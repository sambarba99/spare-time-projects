## PyTorch Knowledge Distillation demo with the CIFAR-10 dataset

|          Model          |  Test set F1 score  |
|:-----------------------:|:-------------------:|
|         Teacher         |        0.805        |
|     Student (no KD)     |        0.761        |
| **_Student (with KD)_** | **0.782** (+ 0.021) |

<p align="center">
	<img src="images/kd_student_output.png"/>
</p>

Teacher architecture (4,532,378 parameters):

<p align="center">
	<img src="images/teacher_architecture.png"/>
</p>

Student architecture (280,218 parameters, 0.06x teacher):

<p align="center">
	<img src="images/student_architecture.png"/>
</p>

Sources:
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531) (Hinton, Vinyals, Dean 2015)
- [Knowledge Distillation Tutorial](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) (PyTorch tutorial)
- [CIFAR-10](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders) (Kaggle dataset)
