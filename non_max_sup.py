import numpy as np

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

def nms(boxes, scores, threshold):
    # score is a list
    score_index = np.argsort(scores)[::-1]
    res =[]
    while len(score_index)>0:
        big_box= boxes[score_index[0]]
        res.append(big_box)
        
        rest_box_index = score_index[1:]
        ious =[ iou(big_box, boxes[index]) for index in rest_box_index]

        # iou > threshold then drop
        remain_index = []
        for i , index in enumerate(rest_box_index):
            if ious[i] > threshold:
                remain_index.append(index )
        score_index= remain_index
        
        #或者 np ： sorted_indices = rest_indices[ious <= threshold]

    return res

# Example usage
boxes = np.array([
    [100, 100, 200, 200],
    [110, 110, 210, 210],
    [300, 300, 400, 400]
])

scores = np.array([0.9, 0.8, 0.7])
threshold = 0.5

output = nms(boxes, scores, threshold)
print("Selected boxes after NMS:", output)