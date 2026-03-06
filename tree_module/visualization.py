import cv2

def draw_detections(image, predictions):

    output = image.copy()

    for _, row in predictions.iterrows():

        xmin = int(row.xmin)
        ymin = int(row.ymin)
        xmax = int(row.xmax)
        ymax = int(row.ymax)

        cv2.rectangle(
            output,
            (xmin, ymin),
            (xmax, ymax),
            (0,255,0),
            2
        )

    return output