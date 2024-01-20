def yolo2coord(x, y, w1, h1, img_w, img_h):
    xmin = round(img_w * (x - w1 / 2.0))
    xmax = round(img_w * (x + w1 / 2.0))
    ymin = round(img_h * (y - h1 / 2.0))
    ymax = round(img_h * (y + h1 / 2.0))
    return xmin, ymin, xmax, ymax
