# Chrismas_AI_app
Cross platform VR app using OpenCV effects, with person segmentation inference served by an external cloud GPU HTTP endpoint.

## Cloud GPU segmentation endpoint
Set these environment variables before running `./app`:

- `CLOUD_GPU_ENDPOINT` (default: `http://localhost:8000/infer`)
- `CLOUD_GPU_TIMEOUT` (seconds, default: `5`)

The app sends each webcam frame as base64-encoded JPEG JSON payload:

```json
{
  "image": "<base64-jpeg>"
}
```

Expected response JSON must contain an 8-bit mask and inverse mask with the **same height and width as the input image** (`255` = human, `0` = background):

```json
{
  "mask": [[0, 255, 255], [0, 0, 255]],
  "mask_inv": [[255, 0, 0], [255, 255, 0]]
}
```

A nested format is also accepted:

```json
{
  "data": {
    "mask": [[0, 255], [255, 0]],
    "mask_inv": [[255, 0], [0, 255]]
  }
}
```
