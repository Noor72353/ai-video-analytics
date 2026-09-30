class LineCrossingAnalyzer:

    def __init__(self, target_classes, line_position):

        self.target_classes = target_classes
        self.line_position = line_position

        self.previous_y = {}

        self.entry_counts = {
            class_id: 0
            for class_id in target_classes
        }

        self.exit_counts = {
            class_id: 0
            for class_id in target_classes
        }

        self.counted_crossings = set()


    def update(
        self,
        track_ids,
        class_ids,
        center_ys,
        line_y,
    ):

        for track_id, class_id, center_y in zip(
            track_ids,
            class_ids,
            center_ys,
        ):

            if track_id in self.previous_y:

                old_y = self.previous_y[track_id]


                # Top -> Bottom = Entry
                if (
                    old_y < line_y
                    and center_y >= line_y
                ):

                    crossing = (
                        track_id,
                        "entry",
                    )

                    if crossing not in self.counted_crossings:

                        self.entry_counts[class_id] += 1

                        self.counted_crossings.add(
                            crossing
                        )


                # Bottom -> Top = Exit
                elif (
                    old_y > line_y
                    and center_y <= line_y
                ):

                    crossing = (
                        track_id,
                        "exit",
                    )

                    if crossing not in self.counted_crossings:

                        self.exit_counts[class_id] += 1

                        self.counted_crossings.add(
                            crossing
                        )


            self.previous_y[track_id] = center_y


    def get_entry_counts(self):

        return self.entry_counts.copy()


    def get_exit_counts(self):

        return self.exit_counts.copy()


    def reset(self):

        self.previous_y.clear()

        self.counted_crossings.clear()

        self.entry_counts = {
            class_id: 0
            for class_id in self.target_classes
        }

        self.exit_counts = {
            class_id: 0
            for class_id in self.target_classes
        }
