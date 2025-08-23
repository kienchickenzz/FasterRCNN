import numpy as np
import os
import random
import xml.etree.ElementTree as ET
from typing import Tuple, Callable

from .training_sample import Box
from .training_sample import TrainingSample
from . import image
from pytorch.FasterRCNN.models import anchors
from .image import PreprocessingParams


class Dataset:
    """
    A dataset iterator for a particular split (train, val, etc.)
    """

    num_classes = 3
    class_index_to_name = {
        0:  "background",
        1:  "dog",
        2:  "cat",
    }

    def __init__( 
        self, 
        split: str, 
        image_preprocessing_params: PreprocessingParams, 
        compute_feature_map_shape_fn, 
        feature_pixels: int = 16, 
        dir: str = "asirra_cat_vs_dogs", 
        augment: bool = True, 
        shuffle: bool = True, 
        allow_difficult: bool = False, 
        cache: bool = True
    ):
        """
        Parameters
        ----------
        split : str
            Dataset split to load: train, val, or trainval.
        image_preprocessing_params : dataset.image.PreprocessingParams
            Image preprocessing parameters to apply when loading images.
        compute_feature_map_shape_fn : Callable[Tuple[int, int, int], Tuple[int, int, int]]
            Function to compute feature map shape, (channels, height, width), from
        input image shape, (channels, height, width).
        feature_pixels : int
            Size of each cell in the Faster R-CNN feature map in image pixels. This
            is the separation distance between anchors.
        dir : str
            Root directory of dataset.
        augment : bool
            Whether to randomly augment (horizontally flip) images during iteration
            with 50% probability.
        shuffle : bool
            Whether to shuffle the dataset each time it is iterated.
        allow_difficult : bool
            Whether to include ground truth boxes that are marked as "difficult".
        cache : bool
            Whether to training samples in memory after first being generated.
        """
        if not os.path.exists( dir ):
            raise FileNotFoundError( "Dataset directory does not exist: %s" % dir )
        self._dir = dir
        
        self.split = split
        
        self.class_index_to_name = self._get_classes()
        assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
        
        self.class_name_to_index = { class_name: class_index for ( class_index, class_name ) in self.class_index_to_name.items() }
        
        self.num_classes = len( self.class_index_to_name )
        assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % ( self.num_classes, Dataset.num_classes )
        
        self._filepaths = self._get_filepaths()
        self.num_samples = len( self._filepaths )
        self._gt_boxes_by_filepath = self._get_ground_truth_boxes( filepaths = self._filepaths, allow_difficult = allow_difficult )
        self._i = 0
        self._iterable_filepaths = self._filepaths.copy()
        self._image_preprocessing_params = image_preprocessing_params
        self._compute_feature_map_shape_fn = compute_feature_map_shape_fn
        self._feature_pixels = feature_pixels
        self._augment = augment
        self._shuffle = shuffle
        self._cache = cache
        self._unaugmented_cached_sample_by_filepath = {}
        self._augmented_cached_sample_by_filepath = {}

    def __iter__( self ):
        self._i = 0
        if self._shuffle:
            random.shuffle( self._iterable_filepaths )
        return self

    def __next__( self ):
        if self._i >= len( self._iterable_filepaths ):
            raise StopIteration

        # Next file to load
        filepath = self._iterable_filepaths[ self._i ]
        self._i += 1

        # Augment?
        flip = random.randint( 0, 1 ) != 0 if self._augment else 0
        cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

        # Load and, if caching, write back to cache
        if filepath in cached_sample_by_filepath:
            sample = cached_sample_by_filepath[ filepath ]
        else:
            sample = self._generate_training_sample( filepath = filepath, flip = flip )
        if self._cache:
            cached_sample_by_filepath[ filepath ] = sample

        # Return the sample
        return sample

    def _generate_training_sample( self, filepath, flip ):
        # Load and preprocess the image
        scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image( 
            url = filepath, 
            preprocessing = self._image_preprocessing_params, 
            min_dimension_pixels = 600, 
            horizontal_flip = flip )
        _, original_height, original_width = original_shape

        # Scale ground truth boxes to new image size
        scaled_gt_boxes = []
        for box in self._gt_boxes_by_filepath[ filepath ]:
            if flip:
                corners = np.array( [
                    box.corners[ 0 ],
                    original_width - 1 - box.corners[ 3 ],
                    box.corners[ 2 ],
                    original_width - 1 - box.corners[ 1 ]
                ] )
            else:
                corners = box.corners
            scaled_box = Box(
                class_index = box.class_index,
                class_name = box.class_name,
                corners = corners * scale_factor
            )
        scaled_gt_boxes.append( scaled_box )

        # Generate anchor maps and RPN truth map
        anchor_map, anchor_valid_map = anchors.generate_anchor_maps( 
            image_shape = scaled_image_data.shape, 
            feature_map_shape = self._compute_feature_map_shape_fn( scaled_image_data.shape ), 
            feature_pixels = self._feature_pixels )
        gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(
            anchor_map = anchor_map, 
            anchor_valid_map = anchor_valid_map, 
            gt_boxes = scaled_gt_boxes )

        # Return sample
        return TrainingSample(
            anchor_map = anchor_map,
            anchor_valid_map = anchor_valid_map,
            gt_rpn_map = gt_rpn_map,
            gt_rpn_object_indices = gt_rpn_object_indices, # type: ignore
            gt_rpn_background_indices = gt_rpn_background_indices, # type: ignore
            gt_boxes = scaled_gt_boxes,
            image_data = scaled_image_data,
            image = scaled_image,
            filepath = filepath
        )

    def _get_classes( self ):
        return {
            0: "background",
            1: "dog", 
            2: "cat",
        }

    def _get_filepaths( self ):
        import glob
        
        cat_images = glob.glob( os.path.join( self._dir, "cat.*.jpg" ) )
        dog_images = glob.glob( os.path.join( self._dir, "dog.*.jpg" ) )
        
        all_images = cat_images + dog_images
        all_images.sort()
        
        # Split dataset into train (80%) and validation (20%)
        if self.split == "train":
            return all_images[ :int( 0.8 * len( all_images ) ) ]
        elif self.split == "val":
            return all_images[ int( 0.8 * len( all_images ) ): ]
        else:
            return all_images

        """
        # Debug: 60 car training images. Handy for quick iteration and testing.
        image_paths = [
            "2008_000028",
            "2008_000074",
            "2008_000085",
            "2008_000105",
            "2008_000109",
            "2008_000143",
            "2008_000176",
            "2008_000185",
            "2008_000187",
            "2008_000189",
            "2008_000193",
            "2008_000199",
            "2008_000226",
            "2008_000237",
            "2008_000252",
            "2008_000260",
            "2008_000315",
            "2008_000346",
            "2008_000356",
            "2008_000399",
            "2008_000488",
            "2008_000531",
            "2008_000563",
            "2008_000583",
            "2008_000595",
            "2008_000613",
            "2008_000619",
            "2008_000719",
            "2008_000833",
            "2008_000944",
            "2008_000953",
            "2008_000959",
            "2008_000979",
            "2008_001018",
            "2008_001039",
            "2008_001042",
            "2008_001104",
            "2008_001169",
            "2008_001196",
            "2008_001208",
            "2008_001274",
            "2008_001329",
            "2008_001359",
            "2008_001375",
            "2008_001440",
            "2008_001446",
            "2008_001500",
            "2008_001533",
            "2008_001541",
            "2008_001631",
            "2008_001632",
            "2008_001716",
            "2008_001746",
            "2008_001860",
            "2008_001941",
            "2008_002062",
            "2008_002118",
            "2008_002197",
            "2008_002202",
            "2011_003247"
        ]
        return [ os.path.join(self._dir, "JPEGImages", path) + ".jpg" for path in image_paths ]
        """

    def _get_ground_truth_boxes( self, filepaths: list[ str ], allow_difficult: bool ):
        gt_boxes_by_filepath = {}

        for filepath in filepaths:
            # Identify XML file
            basename = os.path.splitext( os.path.basename( filepath ) )[ 0 ]
            annotation_file = os.path.join(self._dir, basename ) + ".xml"
            
            # Parse XML file
            tree = ET.parse( annotation_file )
            root = tree.getroot()
            assert tree != None, "Failed to parse %s" % annotation_file
            
            # Validate XML file
            assert len( root.findall( "size" ) ) == 1
            size_obj = root.find( "size" )
            assert size_obj != None, "Failed to parse %s" % annotation_file
            
            assert len( size_obj.findall( "depth" ) ) == 1
            depth_obj = size_obj.find( "depth" )
            assert depth_obj != None, "Failed to parse %s" % annotation_file
            depth = depth_obj.text
            assert depth != None, "Failed to parse %s" % annotation_file
            depth = int( depth )
            assert depth == 3
            
            boxes = []
            for obj in root.findall( "object" ):
                assert len( obj.findall( "name" ) ) == 1
                assert len( obj.findall( "bndbox" ) ) == 1
                assert len( obj.findall( "difficult" ) ) == 1

                # Parse difficult tag
                difficult_obj = obj.find( "difficult" )
                assert difficult_obj != None, "Failed to parse %s" % annotation_file
                is_difficult = difficult_obj.text
                assert is_difficult != None, "Failed to parse %s" % annotation_file

                is_difficult = int( is_difficult ) != 0
                if is_difficult and not allow_difficult:
                    continue # Ignore difficult examples unless asked to include them
                
                # Parse name tag
                class_obj = obj.find( "name" )
                assert class_obj != None, "Failed to parse %s" % annotation_file
                class_name = str( class_obj.text )
                
                # Parse bndbox tag
                bndbox = obj.find( "bndbox" )
                assert bndbox != None, "Failed to parse %s" % annotation_file

                x_min_obj = bndbox.find( "xmin" )
                y_min_obj = bndbox.find( "ymin" )
                x_max_obj = bndbox.find( "xmax" )
                y_max_obj = bndbox.find( "ymax" )
                assert x_min_obj != None, "Failed to parse %s" % annotation_file
                assert y_min_obj != None, "Failed to parse %s" % annotation_file
                assert x_max_obj != None, "Failed to parse %s" % annotation_file
                assert y_max_obj != None, "Failed to parse %s" % annotation_file

                x_min = x_min_obj.text
                y_min = y_min_obj.text
                x_max = x_max_obj.text
                y_max = y_max_obj.text
                assert x_min != None, "Failed to parse %s" % annotation_file
                assert y_min != None, "Failed to parse %s" % annotation_file
                assert x_max != None, "Failed to parse %s" % annotation_file
                assert y_max != None, "Failed to parse %s" % annotation_file

                x_min = int( float( x_min ) ) - 1 # convert to 0-based pixel coordinates
                y_min = int( float( y_min ) ) - 1 # convert to 0-based pixel coordinates
                x_max = int( float( x_max ) ) - 1 # convert to 0-based pixel coordinates
                y_max = int( y_max ) - 1 # convert to 0-based pixel coordinates
                
                corners = np.array( [ y_min, x_min, y_max, x_max ] ).astype( np.float32 )
                box = Box( 
                    class_index = self.class_name_to_index[ class_name ], 
                    class_name = class_name, 
                    corners = corners )
                boxes.append( box )
            
            assert len( boxes ) > 0
            gt_boxes_by_filepath[ filepath ] = boxes
        
        return gt_boxes_by_filepath
