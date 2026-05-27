"""
Smart Home Skill Injection

Injects device states and control functions into the agent's runtime.
"""

from cave_agent.runtime import Function

# Current state of all devices
devices = {
    "living_room_light": {"type": "light", "state": "off", "brightness": 0},
    "bedroom_light": {"type": "light", "state": "off", "brightness": 0},
    "kitchen_light": {"type": "light", "state": "on", "brightness": 80},
    "tv": {"type": "tv", "state": "off", "channel": 5, "volume": 30},
    "front_door": {"type": "door", "state": "locked"},
    "garage_door": {"type": "door", "state": "unlocked"},
}


def set_light(device_id: str, state: str, brightness: int = 100) -> dict:
    """
    Control a light.

    Args:
        device_id: Light device ID (e.g., "living_room_light")
        state: "on" or "off"
        brightness: Brightness level 0-100 (only applies when state is "on")

    Returns:
        Updated device state
    """
    if device_id not in devices:
        return {"error": f"Device '{device_id}' not found"}

    device = devices[device_id]
    if device["type"] != "light":
        return {"error": f"Device '{device_id}' is not a light"}

    device["state"] = state
    device["brightness"] = brightness if state == "on" else 0
    return {"device_id": device_id, **device}


def set_door(device_id: str, state: str) -> dict:
    """
    Control a door lock.

    Args:
        device_id: Door device ID (e.g., "front_door")
        state: "locked" or "unlocked"

    Returns:
        Updated device state
    """
    if device_id not in devices:
        return {"error": f"Device '{device_id}' not found"}

    device = devices[device_id]
    if device["type"] != "door":
        return {"error": f"Device '{device_id}' is not a door"}

    device["state"] = state
    return {"device_id": device_id, **device}


def set_tv(state: str, channel: int = None, volume: int = None) -> dict:
    """
    Control the TV.

    Args:
        state: "on" or "off"
        channel: Channel number (optional)
        volume: Volume level 0-100 (optional)

    Returns:
        Updated device state
    """
    device = devices["tv"]
    device["state"] = state

    if state == "on":
        if channel is not None:
            device["channel"] = channel
        if volume is not None:
            device["volume"] = volume

    return {"device_id": "tv", **device}


def get_status(device_id: str = None) -> dict:
    """
    Get status of devices.

    Args:
        device_id: Specific device ID, or None to get all devices

    Returns:
        Device state or dict of all device states
    """
    if device_id:
        if device_id not in devices:
            return {"error": f"Device '{device_id}' not found"}
        return {"device_id": device_id, **devices[device_id]}
    return devices


__exports__ = [
    Function(set_light, description="Turn a light on/off and set brightness"),
    Function(set_door, description="Lock or unlock a door"),
    Function(set_tv, description="Control the TV power, channel, and volume"),
    Function(get_status, description="Get current status of devices"),
]
