# WarpGradRL
Warp Gradient Descent paper implementation for Maze navigation task

## Instructions to run

To run warped RNN `python warped_maze.py`

To run Hebbian RNN `python hebbian_maze.py`

To run Baseline RNN `python hebbian_maze.py --type rnn`

## Plot
![](https://drive.google.com/file/d/1LqtVqcQj2oRAlN0K4SVAgjvTkdYoKku5/view?usp=sharing)

## References

Thanks Sebastian for clarifying doubts with the code :) 

@article{Flennerhag:2018alstm,
  title   = {{Breaking the Activation Function Bottleneck through Adaptive Parameterization}},
  author  = {Flennerhag, Sebastian and Hujun, Yin and Keane, John and Elliot, Mark},
  journal = {{arXiv preprint arXiv:1805.08574}},
  year    = {2018}
}

@article{Flennerhag2019MetaLearningWW,
  title={Meta-Learning with Warped Gradient Descent},
  author={Sebastian Flennerhag and Andrei A. Rusu and Razvan Pascanu and Hujun Yin and Raia Hadsell},
  journal={ArXiv},
  year={2019},
  volume={abs/1909.00025}
}

@article{Miconi2019BackpropamineTS,
  title={Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity},
  author={Thomas Miconi and Aditya Rawal and Jeff Clune and Kenneth O. Stanley},
  journal={ArXiv},
  year={2019},
  volume={abs/2002.10585}
}

