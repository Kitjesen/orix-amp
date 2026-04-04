import gymnasium as gym

gym.register(
    id="OrixAmp-Isaac-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "config.flat_env_cfg:OrixDogFlatEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="OrixAmp-Isaac-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "config.rough_env_cfg:OrixDogRoughEnvCfg"},
    disable_env_checker=True,
)
