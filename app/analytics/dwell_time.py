import cv2


class DwellTimeAnalyzer:

    def __init__(self):

        self.zone_entry_time = {}


    def update(
        self,
        track_ids,
        inside_states,
    ):

        current_time = cv2.getTickCount()

        frequency = cv2.getTickFrequency()

        dwell_times = {}


        for track_id, inside_zone in zip(
            track_ids,
            inside_states,
        ):

            if inside_zone:

                if track_id not in self.zone_entry_time:

                    self.zone_entry_time[
                        track_id
                    ] = current_time


                elapsed_ticks = (
                    current_time
                    - self.zone_entry_time[track_id]
                )


                elapsed_seconds = (
                    elapsed_ticks / frequency
                )


                dwell_times[track_id] = (
                    elapsed_seconds
                )

            else:

                self.zone_entry_time.pop(
                    track_id,
                    None
                )


        return dwell_times


    def get_dwell_time(self, track_id):

        if track_id not in self.zone_entry_time:

            return 0.0


        current_time = cv2.getTickCount()

        frequency = cv2.getTickFrequency()


        elapsed_ticks = (
            current_time
            - self.zone_entry_time[track_id]
        )


        return elapsed_ticks / frequency


    def reset(self):

        self.zone_entry_time.clear()
