import time
import math

class CooldownManager:
    """
    Manages cooldowns for screenshot events to prevent spamming.
    Tracks events by track_id (primary) and spatial location (fallback).
    """
    def __init__(self, cooldown_seconds=120, spatial_radius=100):
        """
        Args:
            cooldown_seconds (float): Time in seconds before allowing another capture for the same person/type.
            spatial_radius (float): Radius in pixels to consider the same person/location (fallback).
        """
        self.cooldown_seconds = cooldown_seconds
        self.spatial_radius = spatial_radius
        # List of {'track_id': id, 'x': x, 'y': y, 'time': timestamp, 'type': type}
        self.history = []

    def should_capture(self, track_id, x, y, event_type):
        """
        Checks if an event of this type for this track_id or location can be captured.
        Returns True if allowed (cooldown expired or new person), False otherwise.
        """
        self._cleanup()

        for record in self.history:
            if record['type'] == event_type:
                # 1. Check ID Match (Strongest)
                if track_id is not None and record['track_id'] is not None:
                    if track_id == record['track_id']:
                        return False
                    else:
                        # IDs are both present but different.
                        # We assume these are different people.
                        # Skip spatial check for this specific record.
                        continue

                # 2. Check Spatial Match (Fallback if ID is None or missing)
                dist = math.sqrt((x - record['x'])**2 + (y - record['y'])**2)
                if dist < self.spatial_radius:
                    return False

        return True

    def record_capture(self, track_id, x, y, event_type):
        """
        Records an event capture to start the cooldown.
        """
        self.history.append({
            'track_id': track_id,
            'x': x,
            'y': y,
            'time': time.time(),
            'type': event_type
        })

    def _cleanup(self):
        """
        Removes expired records.
        """
        current_time = time.time()
        self.history = [
            rec for rec in self.history
            if (current_time - rec['time']) < self.cooldown_seconds
        ]
