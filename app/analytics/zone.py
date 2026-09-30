class ZoneAnalyzer:

    def __init__(self, target_classes):

        self.target_classes = target_classes

        self.previous_inside = {}

        self.zone_entry_counts = {
            class_id: 0
            for class_id in target_classes
        }

        self.zone_exit_counts = {
            class_id: 0
            for class_id in target_classes
        }


    def is_inside(
        self,
        center_x,
        center_y,
        zone_x1,
        zone_y1,
        zone_x2,
        zone_y2,
    ):

        return (
            zone_x1 <= center_x <= zone_x2
            and
            zone_y1 <= center_y <= zone_y2
        )


    def update(
        self,
        track_ids,
        class_ids,
        centers,
        zone,
    ):

        zone_counts = {
            class_id: 0
            for class_id in self.target_classes
        }


        zone_x1, zone_y1, zone_x2, zone_y2 = zone


        for track_id, class_id, center in zip(
            track_ids,
            class_ids,
            centers,
        ):

            center_x, center_y = center


            inside_zone = self.is_inside(
                center_x,
                center_y,
                zone_x1,
                zone_y1,
                zone_x2,
                zone_y2,
            )


            if inside_zone:

                zone_counts[class_id] += 1


            if track_id in self.previous_inside:

                was_inside = (
                    self.previous_inside[track_id]
                )


                # Outside -> Inside
                if (
                    not was_inside
                    and inside_zone
                ):

                    self.zone_entry_counts[
                        class_id
                    ] += 1


                # Inside -> Outside
                elif (
                    was_inside
                    and not inside_zone
                ):

                    self.zone_exit_counts[
                        class_id
                    ] += 1


            self.previous_inside[
                track_id
            ] = inside_zone


        return zone_counts


    def get_entry_counts(self):

        return self.zone_entry_counts.copy()


    def get_exit_counts(self):

        return self.zone_exit_counts.copy()


    def reset(self):

        self.previous_inside.clear()

        self.zone_entry_counts = {
            class_id: 0
            for class_id in self.target_classes
        }

        self.zone_exit_counts = {
            class_id: 0
            for class_id in self.target_classes
        }
