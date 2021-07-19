

## <div align="center">The paper </div>


<a href="https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> - training 


<a href="https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> - inference  


## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.7.0**](https://www.python.org/) is required with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
```bash
$ git clone https://github.com/leetoo/thepaper
$ cd thepaper
$ pip install -r requirements.txt
```


</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 based model you can execute agains your *_* video / images *_* here :  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

```python
import torch 

# Model
model = ...  # or yolov5m, yolov5x, custom

# Images
img = 'https://... '  # or file, PIL, OpenCV, numpy, multiple

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>


<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources
and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results on [DataSetv5](https://zenodo.org/record/5110223/files/dsv4_img4v_yolo_new_train_3m_c.zip?download=1) dataset (dataset auto-downloads). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data $$$ --cfg $$$ --weights yolov5l --batch-size 64
```

</details>  


<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️ RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW

todo ask do we need those tutorials ? 
</details>


## <div align="center">Environments and Integrations</div>

Get started in seconds with our verified environments and integrations, including [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme) for automatic YOLOv5 experiment logging. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-small.png" width="15%"/>
    </a>
</div>  

## <div align="center">Contact</div>

For issues running the paper please visit [GitHub Issues](https://github.com/leetoo/thepaper/issues). For business or professional support requests please visit 
[https://www.fellowship.ai](https://www.fellowship.ai) 

<br>

<div align="center">
    <a href="https://github.com/fellowship">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="5%" alt=""/>
    </a>
    <img width="5%" />
    <a href="https://www.linkedin.com/company/fellowship-ai/about/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="5%"/>
    </a>
    <img width="5%" />
    <a href="https://twitter.com/fellowshipai">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="5%"/>
    </a>
    <img width="5%" />
    <a href="https://www.youtube.com/channel/UC4VSZUj05MVG-J8n_22fwFQ">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="5%"/>
    </a>
    <img width="5%" />
</div>