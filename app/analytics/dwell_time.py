import time


class DwellTimeAnalyzer:

    def __init__(self, target_classes):
        self.target_classes = target_classes

        # Track the start time of the current visible session.
        self.entry_times = {}

        # Store accumulated dwell time for each track.
        self.accumulated_times = {}

        # Maximum accumulated dwell time observed for each class.
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

            # New track.
            if track_id not in self.entry_times:

                self.entry_times[track_id] = (
                    current_time,
                    class_id,
                )

                if track_id not in self.accumulated_times:
                    self.accumulated_times[track_id] = 0.0

            entry_time, tracked_class_id = (
                self.entry_times[track_id]
            )

            # Current visible-session duration.
            current_session_time = (
                current_time - entry_time
            )

            # Total dwell time for this track.
            total_dwell_time = (
                self.accumulated_times[track_id]
                + current_session_time
            )

            # Keep the maximum accumulated dwell time
            # for the object's class.
            self.dwell_times[
                tracked_class_id
            ] = max(
                self.dwell_times[
                    tracked_class_id
                ],
                total_dwell_time,
            )

        # Objects that disappeared from the current frame.
        inactive_tracks = (
            set(self.entry_times.keys())
            - current_track_ids
        )

        for track_id in inactive_tracks:

            entry_time, tracked_class_id = (
                self.entry_times[track_id]
            )

            # Save the completed visible session.
            session_time = (
                current_time - entry_time
            )

            self.accumulated_times[track_id] += (
                session_time
            )

            # Remove only the active session.
            del self.entry_times[track_id]

        return self.dwell_times.copy()

    def get_dwell_times(self):

        return self.dwell_times.copy()

    def reset(self):

        self.entry_times.clear()

        self.accumulated_times.clear()

        self.dwell_times = {
            class_id: 0.0
            for class_id in self.target_classes
        }
