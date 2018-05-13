# train model
python train_model.py --dataset images --model zoro-notzoro.model --plot plot_zoro.png --resize 28 --architecture LeNet --object Zoro

# test model_single_image
python test_model.py --model zoro-notzoro.model --image examples/zoro1.jpg --resize 28 --object Zoro
python test_model.py --model zoro-notzoro.model --image examples/notzoro1.jpg --resize 28 --object Zoro
python test_model.py --model zoro-notzoro.model --image examples/notzoro_sword1.jpg --resize 28 --object Zoro

 - change the --image based on example images in example/ folder

# test model
python test_model.py --model zoro-notzoro.model --dataset examples --resize 28 --object Zoro

 - change the --object Zoro with any object and corresponding --model name