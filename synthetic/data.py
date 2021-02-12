import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ArithmeticData(Dataset):
	def __init__(self, num_examples, example_length):
		self.num_examples = num_examples
		self.example_length = example_length

		self.digits = []
		self.targets = []
		self.operations = []

		for i in range(self.num_examples):
			target = 0 
			digs = []
			ops =  []
			tar = []
			for i in range(self.example_length):
				if random.uniform(0, 1) > 0:
					digs.append(np.array([round(random.uniform(0, 1), 2)]))
					operation_index = random.randint(0, 2)
					if operation_index == 0:
						target += digs[-1][0]
					elif operation_index == 1:
						target -= digs[-1][0]
					else:
						target *= digs[-1][0]
					op = np.zeros((3))
					op[operation_index] = 1
					ops.append(op)
					tar.append(np.array([target]))
				else:
					digs.append(np.array([0]))
					ops.append(np.array([-1,-1]))
					tar.append(np.array([target]))

			self.targets.append(tar)
			self.digits.append(digs)
			self.operations.append(ops)

	def __len__(self):
		return self.num_examples

	def __getitem__(self, i):
		dig = self.digits[i]
		op = self.operations[i]
		dig = np.stack(dig, axis = 1)
		op = np.stack(op, axis = 1)
		tar = np.stack(self.targets[i], axis = 1)
		inp = np.concatenate((dig, op, tar), axis = 0).astype(np.float)
		return inp

class ArithmeticDataSpec(Dataset):
	def __init__(self, num_examples, example_length):
		self.num_examples = num_examples
		self.example_length = example_length

		self.digit_1 = []
		self.digit_2 = []
		self.operation = []
		self.target = []

		for i in range(self.num_examples):
			range_index = random.randint(0, 2)
			if range_index == 0:
				self.digit_1.append(np.array([random.uniform(0, 0.33)]))
				self.digit_2.append(np.array([random.uniform(0, 0.33)]))
				self.operation.append(np.array([1.,0.,0.]))
				self.target.append(self.digit_1[-1] + self.digit_2[-1])
			elif range_index == 1:
				self.digit_1.append(np.array([random.uniform(0.33, 0.66)]))
				self.digit_2.append(np.array([random.uniform(0.33, 0.66)]))
				self.operation.append(np.array([0.,1.,0.]))
				self.target.append(self.digit_1[-1] - self.digit_2[-1])
			else:
				self.digit_1.append(np.array([random.uniform(0.66, 1.0)]))
				self.digit_2.append(np.array([random.uniform(0.66, 1.0)]))
				self.operation.append(np.array([0.,0.,1.]))
				self.target.append(self.digit_1[-1] * self.digit_2[-1])

		"""for i in range(self.num_examples):
			target = 0 
			digs = []
			ops =  []
			tar = []
			for i in range(self.example_length):
				if random.uniform(0, 1) > 0:
					range_index = random.randint(0, 2)
					if range_index == 0:
						digs.append(np.array([round(random.uniform(0, 0.33), 2)]))
						op = np.zeros((3))
						op[0] = 1
						ops.append(op)
						target += digs[-1][0]
						tar.append(np.array([target]))
					elif range_index == 1:
						digs.append(np.array([round(random.uniform(0.33, 0.66), 2)]))
						op = np.zeros((3))
						op[1] = 1
						ops.append(op)
						target -= digs[-1][0]
						tar.append(np.array([target]))
					else:
						digs.append(np.array([round(random.uniform(0.66, 1), 2)]))
						op = np.zeros((3))
						op[2] = 1
						ops.append(op)
						target *= digs[-1][0]
						tar.append(np.array([target]))		
				else:
					digs.append(np.array([0]))
					ops.append(np.array([-1,-1]))
					tar.append(np.array([target]))
			self.targets.append(tar)
			self.digits.append(digs)
			self.operations.append(ops)"""

	def __len__(self):
		return self.num_examples

	def __getitem__(self, i):
		
		return self.digit_1[i], self.digit_2[i], self.operation[i], self.target[i]



if __name__ == '__main__':
	data = ArithmeticData(500, 10)
	for i in range(len(data)):
		print(data[i])
		print('----------------') 

