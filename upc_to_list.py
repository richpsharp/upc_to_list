"""Scan default webcam for barcodes and print to terminal/take a pic."""
import logging

from pyzbar.pyzbar import decode
import cv2
import numpy
import shapely.geometry
import shapely.ops

N_TO_BUFFER = 5  # see for this many frames before you're sure

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    last_data = None
    polygon_list = []
    while True:
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            break
        ret, frame = cam.read()
        if not ret:
            continue
        cv2.imshow("UPC searcher", frame)
        try:
            decode_result = decode(frame)
            if not decode_result:
                last_data = None
                continue
            if decode_result[0].data != last_data:
                last_data = decode_result[0].data
                polygon_list = []

            if len(polygon_list) == N_TO_BUFFER:
                # don't keep scanning if we got it
                continue

            poly = decode_result[0].polygon
            if len(poly) <= 2:
                continue
            polygon_list.append(
                shapely.geometry.Polygon(numpy.array([(p.x, p.y) for p in poly])))
            if len(polygon_list) < N_TO_BUFFER:
                continue
            bounding_poly = shapely.ops.cascaded_union(polygon_list)
            point1 = (int(v) for v in bounding_poly.bounds[0:2])
            point2 = (int(v) for v in bounding_poly.bounds[2:4])
            rectangle = [
                int(bounding_poly.bounds[0]),
                int(bounding_poly.bounds[1]),
                int(bounding_poly.bounds[2])-int(bounding_poly.bounds[0]),
                int(bounding_poly.bounds[3])-int(bounding_poly.bounds[1])]
            frame = cv2.rectangle(
                img=frame, rec=rectangle, color=(255, 0, 0), thickness=2)
            frame, cv2.putText(
                frame,
                decode_result[0].data.decode('utf-8'), (5, frame.shape[0]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Found UPC", frame)
            LOGGER.info(f'Found: {decode_result[0].data}')
        except Exception:
            LOGGER.exception('something bad happened')

    cam.release()
    cv2.destroyAllWindows()
