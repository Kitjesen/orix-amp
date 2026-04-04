from isaaclab.utils import configclass
from .rough_env_cfg import OrixDogRoughEnvCfg


@configclass
class OrixDogFlatEnvCfg(OrixDogRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "OrixDogFlatEnvCfg":
            self.disable_zero_weight_rewards()
