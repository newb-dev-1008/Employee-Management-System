# Employee Management System

A resource-friendly face recognition attendance system for employees, with separate modules for training, augmentation and recognition.

### Salient Features
1. Support to host system on a Flask server.
1. Implicit data augmentation.
1. Small, yet powerful models for feature extraction and classifiers with support for tuning hyperparameters as per dataset.
1. Ease of adding new employees / removing old employees and re-training classifiers.

## How it works

### File structure
```
├───backend_support
│   ├───Models
│   │   ├───Embedder
│   │   │       openface_nn4.small2.v1.t7
│   │   │
│   │   └───Face Detection
│   │           deploy.prototxt
│   │           res10_300x300_ssd_iter_140000.caffemodel
│   │
│   └───Scripts
│       │   netSystem.py
│       │   recognizer (Not Required).py
│       │   trainer.py
│
├───Datasets
│   ├───Employee_Name_or_ID_1
│   │
│   ├───Employee_Name_or_ID_2
│   │
│   └───Employee_Name_or_ID_3
│
└───server_app
    │   app.py
```
* `backend_support`: Contains models for face detection and feature extraction, and scripts that utilize them for training and recognition. Also houses embeddings and labels of faces after model is trained.
* `Datasets`: Contains images of all employees in separate folders - each folder containing images of one individual only.
* `server_app`: Contains a Python script to run a server application using the Flask web framework.

### Working principle
1. All images are first routed through a series of functions to
    * perform face localization from the rest of image
    * use facial landmarks for face alignment on all images
    * augment all faces with different filters, skews and orientations

2. The pre-processed data is passed through [OpenFace](https://cmusatyalab.github.io/openface/)'s CNN - a Torch implementation which generates 128-d representations of each image.

3. These 128-d embeddings are pickled together for each person with their respective labels, and are used for training an SVM.

### Guidelines on data collection
For best results, please try and ensure your dataset adheres to the following:

1. Ensure the face occupies at least 60% of the net image area.
2. To increase adaptability, ensure your dataset has images with a variety of expressions - and with a variety of add-ons (spectacles, make-up, etc.)
3. Ensure that the face is not occluded, and main features like the eyes are visibly distinct and are not covered by sunglasses.
4. Ensure that the images pertaining to a person contain the face of the specific person only. Multiple faces in an image can throw the model off.
5. Ensure all images are stored in folders as shown in the Usage section.

## Usage

### For training on new faces
1. Add photos of your employees to the `Datasets` folder, in the following manner:
    * JPEG, PNG and BMP are accepted formats.
    * All employees need not have the same number of images. However, ensure they have unique names or unique IDs for their respective folder names.
    * For best results, please see [Guidelines on data collection](https://github.com/newb-dev-1008/Employee-Management-System?tab=readme-ov-file#guidelines-on-data-collection).

    ```
    Datasets
       ├───Employee_Name_or_ID_1
       │       Employee_ID_1_Image_1.jpeg
       │       Employee_ID_1_Image_2.jpeg
       │       Employee_ID_1_Image_3.jpeg
       │
       ├───Employee_Name_or_ID_2
       │       Employee_ID_2_Image_1.jpeg
       │       Employee_ID_2_Image_2.jpeg
       │       Employee_ID_2_Image_3.jpeg
       │
       └───Employee_Name_or_ID_3
               Employee_ID_3_Image_1.jpeg
               Employee_ID_3_Image_2.jpeg
               Employee_ID_3_Image_3.jpeg
    ```

2. Run `trainer.py` to train the classifier on your set of images.
    * Note that your images will be deleted from the `\Datasets` folder after training, in order to save space. To disable this feature and retain your dataset, set this line in `trainer.py` to `False`:

    ```py
    deleteDataset = False
    ```

3. New employees can be appended to the model in a similar fashion.
    * **Works even if the older employee datasets have been deleted.** However, ensure their embeddings `(/backend_support/Embeddings and labels\Individual Embeddings)` still exist.

    ```
    Datasets
       ├───Old_Employee_1
       │   .
       │   .
       │
       ├───Old_Employee_99
       │       Employee_ID_99_Image_1.jpeg
       │       Employee_ID_99_Image_2.jpeg
       │       Employee_ID_99_Image_3.jpeg
       │
       └───New_Employee_100
               Employee_ID_100_Image_1.jpeg
               Employee_ID_100_Image_2.jpeg
               Employee_ID_100_Image_3.jpeg
    ```

### For running the recognition module
Run `app.py` on localhost or on any production server. Post the image of the employee to the URL, and receive response with name/ID of the identified individual.
