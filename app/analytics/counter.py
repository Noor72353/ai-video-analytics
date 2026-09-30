class ObjectCounter:

    def __init__(self, target_classes):
        self.target_classes = target_classes

        self.counts = {
            class_id: 0
            for class_id in target_classes
        }

    def update(self, detected_class_ids):

        for class_id in detected_class_ids:

            if class_id in self.counts:
                self.counts[class_id] += 1

    def get_counts(self):
        return self.counts.copy()

    def reset(self):

        self.counts = {
            class_id: 0
            for class_id in self.target_classes
        }
