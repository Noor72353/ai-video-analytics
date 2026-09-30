class ObjectTracker:

    def __init__(
        self,
        model,
        tracker_config="bytetrack.yaml",
    ):

        self.model = model
        self.tracker_config = tracker_config


    def track(
        self,
        frame,
        classes=None,
    ):

        return self.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker=self.tracker_config,
            classes=classes,
        )
