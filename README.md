Abstract 
In video surveillance scenarios, face recognition accuracy is often affected by low-resolution and occlusion in captured face images. However, existing face image recovery models typically only handle one type of variation. To address this limitation, we propose a two-step approach that utilizes a Deep Convolutional Generative Adversarial Network (DCGAN) for face completion and a Super Resolution Generative Adversarial Network (SRGAN) for enhancing image resolution. The generator model aims to generate ahigh-resolution face image without occlusion by taking a low-resolution face image with occlusion as input.

Due to the increasing number of thefts recorded by CCTV cameras in public places, it is crucial to recognize faces in surveillance footage. To achieve this, we propose using a cascade classifier to extract frames from the video containing face images. These frames are then fed into an SRGAN to generate high-resolution images. Additionally, in cases where the surveillance camera captures a face in a side view, we propose using a conditional GAN to generate a frontal face view.

Procedure to runand

1) Clone the repository
2) Download the Pix2Pix GAN trained model from Google drive link and put it under Final-Project-Folder https://drive.google.com/drive/folders/1NvfVkzQEsORsEJhvDr3w_7qNI007jeM0?usp=share_link
To create a virtual environment in Miniconda using a `requirements.txt` file, you can follow these steps:
3) create virtual environment
    1. First, make sure you have Miniconda installed on your system. If not, you can download and install it from the official Miniconda website              (https://docs.conda.io/en/latest/miniconda.html) based on your operating system.

    2. Open a terminal or command prompt and execute the following command to create a new virtual environment:

   ```shell
   conda create -n myenv
   ```

   Replace `myenv` with the desired name for your virtual environment.

    3. Activate the newly created virtual environment by running the appropriate command for your operating system:

    - **On Windows:**

     ```shell
     conda activate myenv
     ```


    4. Once the environment is activated, navigate to the directory where your `requirements.txt` file is located using the `cd` command. For example:

    ```shell
    cd /path/to/requirements.txt
    ```

    5. Install the required packages from `requirements.txt` using the `pip` command:

    ```shell
    pip install -r requirements.txt
    ```

    This command will install all the packages listed in the `requirements.txt` file into your virtual environment.

    6. After the installation completes, you can start using your virtual environment with the installed packages.

    Remember to replace `/path/to/requirements.txt` with the actual path to your `requirements.txt` file.

    That's it! You have created a virtual environment in Miniconda and installed the packages specified in the `requirements.txt` file.
