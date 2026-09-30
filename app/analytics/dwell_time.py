import time


class DwellTimeAnalyzer:

    def __init__(self, target_classes):
        self.target_classes = target_classes

        self.entry_times = {}

        self.dwell_times = {
            class_id: 0.0
            for class_id in target_classes
        }

    def update(
        self,
        track_ids,
        class_ids,
        centers,
    ):
        current_time = time.time()

        current_track_ids = set()

        for track_id, class_id, center in zip(
            track_ids,
            class_ids,
            centers,
        ):

            current_track_ids.add(track_id)

            if track_id not in self.entry_times:

                self.entry_times[track_id] = (
                    current_time,
                    class_id,
                )

            entry_time, tracked_class_id = (
                self.entry_times[track_id]
            )

            dwell_time = (
                current_time - entry_time
            )

            self.dwell_times[
                tracked_class_id
            ] = max(
                self.dwell_times[
                    tracked_class_id
                ],
                dwell_time,
            )

        # Remove tracks that are no longer visible.
        inactive_tracks = (
            set(self.entry_times.keys())
            - current_track_ids
        )

        for track_id in inactive_tracks:

            del self.entry_times[track_id]

        return self.dwell_times.copy()

    def get_dwell_times(self):

        return self.dwell_times.copy()

    def reset(self):

        self.entry_times.clear()

        self.dwell_times = {
            class_id: 0.0
            for class_id in self.target_classes
        }
