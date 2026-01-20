import time
import math

class CooldownManager:
    """
    Manages cooldowns for screenshot events to prevent spamming.
    Tracks events by spatial location and type.
    """
    def __init__(self, cooldown_seconds=120, spatial_radius=100):
        """
        Args:
            cooldown_seconds (float): Time in seconds before allowing another capture for the same person/type.
            spatial_radius (float): Radius in pixels to consider the same person/location.
        """
        self.cooldown_seconds = cooldown_seconds
        self.spatial_radius = spatial_radius
        # List of {'x': x, 'y': y, 'time': timestamp, 'type': type}
        self.history = []

    def should_capture(self, x, y, event_type):
        """
        Checks if an event of this type at this location can be captured.
        Returns True if allowed (cooldown expired or new location), False otherwise.
        """
        self._cleanup()

        for record in self.history:
            if record['type'] == event_type:
                # Check distance
                dist = math.sqrt((x - record['x'])**2 + (y - record['y'])**2)
                if dist < self.spatial_radius:
                    # Found a recent event of same type at this location
                    return False
        return True

    def record_capture(self, x, y, event_type):
        """
        Records an event capture to start the cooldown.
        """
        self.history.append({
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
