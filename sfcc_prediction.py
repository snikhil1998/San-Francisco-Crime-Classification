import ctypes
import importlib

CUDA_SUCCESS = 0

libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll', 'cuda.dll')
for libname in libnames:
	try:
		cuda = ctypes.CDLL(libname)
	except OSError:
		continue
	else:
		cuda_flag = True
		break
else:
	cuda_flag = False
	print("could not load any of: " + ' '.join(libnames))

if cuda_flag:
	nGpus = ctypes.c_int()
	result = ctypes.c_int()
	error_str = ctypes.c_char_p()
	result = cuda.cuInit(0)
	if result != CUDA_SUCCESS and cuda_flag:
		cuda.cuGetErrorString(result, ctypes.byref(error_str))
		print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
		cuda_flag = False
	result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
	if result != CUDA_SUCCESS and cuda_flag:
		cuda.cuGetErrorString(result, ctypes.byref(error_str))
		print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
		cuda_flag = False
	cuda_flag = nGpus.value>0

if cuda_flag and importlib.util.find_spec('cupy')!=None and importlib.util.find_spec('cudf')!=None and importlib.util.find_spec('cuml')!=None:
	import sfcc_helper_gpu as sfcc_helper
else:
	import sfcc_helper_cpu as sfcc_helper


if __name__=="__main__":
	#train_path = '../input/train.csv'
	#test_path = '../input/test.csv'
	train_path = 'train.csv'
	test_path = 'test.csv'
	model = sfcc_helper.Models(train_path, test_path)
	model.RandomForestModel()
