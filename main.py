
from io import BytesIO

import gradio as gr
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model import predict_image

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="MRI image classification using MobileNetV3-Large.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "MobileNetV3-Large"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file.")

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        image = Image.open(BytesIO(contents))
        image.load()
        result = predict_image(image)

        return result

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        ) from None


def predict_for_demo(image):
    if image is None:
        return "Please upload an MRI image.", {}

    result = predict_image(image)

    return (
        result["predicted_class"],
        {
            name: f"{score:.2f}%"
            for name, score in result["probabilities"].items()
        },
    )


with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown(
        """
        # Brain Tumor MRI Classifier
        Upload an MRI image to see the model's predicted class.

        **Classes:** Glioma · Meningioma · No Tumor · Pituitary
        """
    )

    image_input = gr.Image(
        type="pil",
        label="Upload MRI image",
    )

    predict_button = gr.Button("Classify MRI", variant="primary")

    prediction_output = gr.Textbox(
        label="Predicted class",
        interactive=False,
    )

    probability_output = gr.JSON(label="Class probabilities")

    predict_button.click(
        fn=predict_for_demo,
        inputs=image_input,
        outputs=[prediction_output, probability_output],
    )

    gr.Markdown(
        """
        **Research demonstration only.** This model is not a medical
        diagnostic tool. Predictions may be incorrect and must not be
        used to make clinical decisions.
        """
    )


app = gr.mount_gradio_app(app, demo, path="/")
