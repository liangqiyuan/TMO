"""Hardware and cloud-service profiles used by ``tmo.env.M4A1_Env``.

This module is the single place where users adapt TMO to a *different physical
deployment*. It contains two registries -- :data:`LOCAL_DEVICES` and
:data:`CLOUD_SERVERS` -- together with the small set of helpers that the
environment uses to turn a profile name into concrete latency / monetary cost
numbers.

The defaults below reproduce exactly the local devices and cloud networks used
in the M4A1 / TMO paper. They are intended as **examples**: external users
adapting TMO to their own scenario are expected to either (a) extend the two
registries with extra entries or (b) replace them with their own dictionary
when calling :func:`compute_local_costs` / :func:`get_cloud_costs`.

Schema
------
A *local-device* profile contributes the device's peak compute throughput
(GFLOPS) and peak power draw (Watts). The local inference time and energy
cost are derived analytically from a small set of LLM-compute parameters
(parameter count, prompt / total length) and the local electricity price.

A *cloud-server* profile contributes two arrays -- ``cloud_time`` and
``cloud_usage_cost`` -- of length ``num_modalities + 1``.  Index ``k`` is the
end-to-end latency / dollar cost of a cloud call that uploads exactly ``k``
modality artefacts (so index ``0`` is the text-only case).
"""

# ---------------------------------------------------------------------------
# Local devices
# ---------------------------------------------------------------------------

LOCAL_DEVICES = {
    "Raspberry Pi-4B":   {"GF_peak": 13.5,   "P_max": 8},
    "Raspberry Pi-5":    {"GF_peak": 31.4,   "P_max": 12},
    "Jetson Nano":       {"GF_peak": 472.0,  "P_max": 10},
    "Jetson TX2":        {"GF_peak": 1.33e3, "P_max": 15},
    "Jetson Xavier NX":  {"GF_peak": 21e3,   "P_max": 20},
    "Jetson Orin NX":    {"GF_peak": 100e3,  "P_max": 25},
    "Jetson AGX Orin":   {"GF_peak": 275e3,  "P_max": 60},
    "iPhone 15 Pro":     {"GF_peak": 35e3,   "P_max": 15},
}

DEFAULT_LLM_COMPUTE = {
    "num_parameter": 3.8e9,
    "L_base": 1024,
    "L_total": 2048,
}

DEFAULT_ELECTRICITY = {"cents_per_kWh": 16.68}


def compute_local_costs(local_device, llm_compute=None, electricity=None,
                        registry=None):
    """Return ``(local_time_seconds, local_usage_cost_dollars)`` for one local call"""
    registry = registry if registry is not None else LOCAL_DEVICES
    if local_device not in registry:
        raise ValueError(
            f"Unknown local device {local_device!r}. "
            f"Available: {sorted(registry.keys())}. "
            f"Add your device to tmo.devices.LOCAL_DEVICES (or pass a custom "
            f"`registry`) to make it available."
        )
    spec = registry[local_device]
    GF_peak = spec["GF_peak"]
    P_max = spec["P_max"]

    llm_compute = llm_compute or DEFAULT_LLM_COMPUTE
    electricity = electricity or DEFAULT_ELECTRICITY
    cost_per_joule = (electricity["cents_per_kWh"] / 100) / (3600 * 1000)

    flops_per_token = 2 * llm_compute["num_parameter"] / llm_compute["L_base"]
    total_flops = flops_per_token * llm_compute["L_total"]
    local_time = total_flops / (GF_peak * 1e9)
    energy = P_max * local_time
    local_usage_cost = energy * cost_per_joule
    return local_time, local_usage_cost


# ---------------------------------------------------------------------------
# Cloud servers
# ---------------------------------------------------------------------------
#
# Each entry contains two arrays that must have length ``num_modalities + 1``.
# The defaults here correspond to the M4A1 setup with ``num_modalities = 3``
# (so each array has 4 entries, indexed by the number of uploaded images).
#
# When adapting TMO to a different ``num_modalities`` you must either:
#   * provide arrays of length ``num_modalities + 1`` here, or
#   * pass a ``registry=`` dict to :func:`get_cloud_costs` at construction.

CLOUD_SERVERS = {
    "Wired": {
        "cloud_time":       [6.46030, 4.76134, 6.24569, 6.98211],
        "cloud_usage_cost": [0.00049, 0.00500, 0.00945, 0.01368],
    },
    "WiFi": {
        "cloud_time":       [0.5, 1.898, 2.016, 2.238],
        "cloud_usage_cost": [0.00049, 0.00500, 0.00945, 0.01368],
    },
    "5G": {
        "cloud_time":       [0.5, 6.037, 6.412, 7.118],
        "cloud_usage_cost": [0.00049, 0.00500, 0.00945, 0.01368],
    },
    "4G": {
        "cloud_time":       [0.5, 16.603, 17.636, 19.573],
        "cloud_usage_cost": [0.00049, 0.00500, 0.00945, 0.01368],
    },
}


def get_cloud_costs(cloud_server, num_modalities, registry=None):
    """Return ``(cloud_time, cloud_usage_cost)`` lists of length ``num_modalities + 1``"""
    registry = registry if registry is not None else CLOUD_SERVERS
    if cloud_server not in registry:
        raise ValueError(
            f"Unknown cloud server {cloud_server!r}. "
            f"Available: {sorted(registry.keys())}. "
            f"Add your network profile to tmo.devices.CLOUD_SERVERS (or pass "
            f"a custom `registry`) to make it available."
        )
    spec = registry[cloud_server]
    cloud_time = list(spec["cloud_time"])
    cloud_usage_cost = list(spec["cloud_usage_cost"])

    expected = num_modalities + 1
    if len(cloud_time) != expected or len(cloud_usage_cost) != expected:
        raise ValueError(
            f"Cloud profile {cloud_server!r} provides cloud_time of length "
            f"{len(cloud_time)} and cloud_usage_cost of length "
            f"{len(cloud_usage_cost)}, but num_modalities={num_modalities} "
            f"requires both arrays to have length {expected} (one entry per "
            f"possible upload count, including 0)."
        )
    return cloud_time, cloud_usage_cost
