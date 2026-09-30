class TrajectoryAnalyzer:

    def __init__(self, max_length=30):

        self.max_length = max_length

        self.trajectories = {}


    def update(
        self,
        track_ids,
        centers,
    ):

        for track_id, center in zip(
            track_ids,
            centers,
        ):

            if track_id not in self.trajectories:

                self.trajectories[track_id] = []


            self.trajectories[track_id].append(
                center
            )


            if (
                len(self.trajectories[track_id])
                > self.max_length
            ):

                self.trajectories[track_id].pop(0)


    def get_trajectory(self, track_id):

        return self.trajectories.get(
            track_id,
            []
        )


    def get_all_trajectories(self):

        return self.trajectories.copy()


    def reset(self):

        self.trajectories.clear()
