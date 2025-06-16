Get AI model from

```
git clone https://huggingface.co/yaya36095/ai-image-detector
```

and show it in app.py as:

```
processor = ViTImageProcessor.from_pretrained("C:/Users/User/Desktop/ai-image-detector")
model = ViTForImageClassification.from_pretrained("C:/Users/User/Desktop/ai-image-detector")
```

To install all requirements:

```
pip install -r requirements.txt
```

To start the projetc:

```
python ./app.py runserver
```
