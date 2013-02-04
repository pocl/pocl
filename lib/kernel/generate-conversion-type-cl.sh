# this script autogenerates the opencl type conversion functions

python convert_type.py --use-fenv yes > x86_64/convert_type.cl
python convert_type.py --use-fenv no  > convert_type.cl
