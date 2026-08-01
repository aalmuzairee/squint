# Squint

<p align="center">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/reach_cube.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/reach_can.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/lift_cube.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/lift_can.gif">
<br>
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/place_cube.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/place_can.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/stack_cube.gif">
<img width="24%" src="https://aalmuzairee.github.io/papers-media/squint/extras/gifs/stack_can.gif">
</p> 

**Fast Visual Reinforcement Learning for Sim-to-Real Robotics**

Squint is a visual Soft Actor Critic method, that through careful image preprocessing, architectural design choices, and hyperparameter selection, is able to leverage parallel environments and experience reuse effectively, achieving faster wall-clock training time than both prior visual off-policy and on-policy methods, and *solving visual tasks in minutes*.

Pytorch Implementation for [[Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics]](https://arxiv.org/abs/2602.21203) by

[Abdulaziz Almuzairee](https://aalmuzairee.github.io) and [Henrik I. Christensen](https://hichristensen.com) (UC San Diego)</br>


[[Website]](https://aalmuzairee.github.io/squint) [[Paper]](https://arxiv.org/abs/2602.21203) 

If you use this code in your research, kindly cite:

```bibtex
@article{almuzairee2026squint,
      title={Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics}, 
      author={Almuzairee, Abdulaziz and Christensen, Henrik I.},
      journal={arXiv preprint arXiv:2602.21203},
      year={2026}
}
```

-----

## 📋 Requirements

- **GPU**: NVIDIA RTX 3080 or better (At least 10GB GPU RAM)
- **Robot**: SO-101 robot arm and wrist camera (Our robot and wrist camera are from [WowRobo](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1))

## 🛠️ Installation

### Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate squint
```

## 🎮 Simulation Training

### Basic Training

Train an agent on the LiftCube task:

```bash
python train_squint.py --env_id=SO101LiftCube-v1
```

### Training with Weights & Biases Logging

We use wandb [(weights and biases)](https://wandb.ai/) for logging, uploading saved models, and downloading them. 
We recommend creating a wandb account, and then enabling `--track` flag and filling the `--wandb_entity` flag in `train_squint.py`. Or you can override with the commandline:

```bash
python train_squint.py \
    --env_id=SO101LiftCube-v1 \
    --track \
    --wandb_entity=YOUR_WANDB_USERNAME
```

At the end of training, the last checkpoint saved will be uploaded to wandb. You can download the last uploaded checkpoint and continue training on it by setting the `--checkpoint=wandb` flag.

### Visualize Environments

You can visualize all available environments (8 environments) by running:

```bash
python examples/visualize_sim.py
```

### Available Environments (SO-101 Task Set)

| Environment | Description | Time to Training Convergence |
|-------------|-------------|-------------|
| `SO101ReachCube-v1` | Reach to a target cube position | 2 minutes |
| `SO101ReachCan-v1` | Reach to a target can position | 2 minutes |
| `SO101LiftCube-v1` | Pick up and lift a cube | 3 minutes |
| `SO101LiftCan-v1` | Pick up and lift a can | 4 minutes |
| `SO101PlaceCube-v1` | Pick up a cube and place in the bin | 5 minutes |
| `SO101PlaceCan-v1` | Pick up a can and place in the bin | 6 minutes |
| `SO101StackCube-v1` | Stack the smaller cube on the larger one | 6 minutes |
| `SO101StackCan-v1` | Stack the cube on the can | 9 minutes |

For all our experiments we train with `--total_timesteps=1_500_000` which takes approximately 15 minutes. You can reduce the number of total timesteps depending on the task. For example, in Reach tasks you can run with `--total_timesteps=200_000` which will take ~2 minutes. Make sure your Squint agent achieves high success rate in simulation before deploying to your real SO-101 robot arm.

### Domain Randomization

All environments have domain randomization implemented to help sim-to-real transfer. There are shared domain randomization parameters between all environments in [`envs/base_random_env.py`](envs/base_random_env.py#L53) and per-task domain randomization 
parameters in each environment file in `envs/`. Feel free to tune these parameters to your real world robot setup.

### Expected Results

For expected results, we show the plots of training with Squint agents below:

</br>
<img width="100%" src="https://aalmuzairee.github.io/papers-media/squint/extras/imgs/per_task_results.png">
</br>


## 🤖 Deployment on Real SO-101 Robot

### Prerequisites

- SO-101 robot arm is functional 
- Wrist camera mounted appropriately
- Calibrated motors using [LeRobot calibration](https://huggingface.co/docs/lerobot/en/so101)

### Step 0: (Optional) Print 3D Objects

We provide the stl files for all 3D objects used in our tasks in `deploy_utils/blender_stls`. 
If you have access to a 3D printer, you should be able to print them, preferably with the following PLA colors:

     bin.stl : white
     can.stl : blue
     cube.stl: red
     large_cube.stl: blue

If you have these objects in different colors, you can alter the colors of these objects in simulation in each
of the tasks to match the real world objects.

### Step 1: Configure Your Robot

Edit `deploy_utils/robot_config.py` with your hardware settings. 

### Step 2: Tune Camera Alignment 

Visual reinforcement learning agents are sensitive to slight visual changes. The more we reduce the difference, the better your agent will transfer. 
We use a table with a black background. In ManiSkill3 simulation, we segment the objects of interest and replace the background with the image 
provided in `envs/black_overlay.png`. Below, we show a visual of the Simulation Env, the Overlay Image (`black_overlay.png`), the Simulation Env with the Overlay in the background, and the Real World Input Image:
</br></br>
<img width="100%" src="https://aalmuzairee.github.io/papers-media/squint/extras/imgs/overlay_example.png">
</br>

If your table has a different background or color, take a photo, save it, and then edit the Randomization Config in [`envs/base_random_env.py`](envs/base_random_env.py#L59) to point to your image. Once you have your image in the background, align your real camera view with the simulation:

```bash
python deploy_utils/tune_camera.py
```
</br>
<img width="100%" src="https://aalmuzairee.github.io/papers-media/squint/extras/imgs/tune_camera_with_box_example.png">
</br>

Adjust the trackbars such that the **gripper and base positions** (outlined in the blue square) in both the simulation and the real world are as close as possible. Once they appear to match, press `p` to print 
the wrist camera parameters, and then copy these parameters straight to wrist camera parameters in [`envs/base_random_env.py`](envs/base_random_env.py#L497)


### Step 3: Deploy

Run your trained agent on the real robot:

```bash
python deploy.py \
    --checkpoint=path/to/ckpt.pt \
    --env_id=SO101LiftCube-v1
```

If you trained with wandb, your last checkpoint should have been uploaded to wandb. You can deploy it by running:

```bash
python deploy.py \
    --checkpoint=wandb \
    --env_id=SO101LiftCube-v1 \
    --wandb_entity=YOUR_WANDB_USERNAME
```

**Keyboard Controls During Deployment:**
- `s` - Skip current episode
- `q` - Quit evaluation

### Deployment Tips

- For safety, run the first run with `--no-continuous_eval`, which will query you for input before each step. If the robot moves reasonably, then you can run without it.
- Test with `--env_id=SO101ReachCube-v1` or `--env_id=SO101ReachCan-v1` before manipulation tasks.
- If at any time during deployment you need to stop, you can press `q` or `ctrl+c`.
- For best performances, run the robot in a well lit room with no sunlight.
- For better transfer from sim to real, make sure the robot motor calibration is good, and the visual alignment between sim and real is good.

## 📁 Project Structure

```
squint/
├── train_squint.py          # Main training script
├── deploy.py                # Real robot deployment
├── utils.py                 # Training utilities
├── environment.yaml         # Conda environment
├── envs/                    # Custom ManiSkill environments
│   ├── base_random_env.py   # Base env with domain randomization
│   ├── black_overlay.png    # Background overlay for sim-to-real
│   ├── reach.py
│   ├── lift.py
│   ├── place.py
│   ├── stack.py
│   └── robot/               # Robot URDF and meshes
├── results/                 # Training results (CSV files per task)
├── examples/
│   └── visualize_sim.py     # Visualize all environments
└── deploy_utils/
    ├── robot_config.py      # Robot hardware config
    ├── manipulator.py       # Real robot interface
    └── tune_camera.py       # Camera alignment tool
```

## 🙏 Acknowledgments

This work would not have been possible without the awesome open source community below:

- [LeanRL](https://github.com/meta-pytorch/LeanRL)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [ManiSkill3](https://github.com/haosulab/ManiSkill)
- [LeRobot Sim2Real ManiSkill3](https://github.com/StoneT2000/lerobot-sim2real)
- [FastTD3](https://github.com/younggyoseo/FastTD3)
- [FastSAC](https://github.com/amazon-far/holosoma)
- [LeRobot](https://github.com/huggingface/lerobot)

We would also like to thank [@jackvial](https://github.com/jackvial) for setting up initial support for [SO-101 Robot Arm in ManiSkill3](https://github.com/StoneT2000/lerobot-sim2real/pull/18)

## 📄 License

This project is [MIT Licensed](LICENSE). Dependencies are subject to their own licenses.

