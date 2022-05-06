mpirun -np 9 -hostfile ./mpi_host_file python3 ./main_fedgkt.py \
--gpu 0 \
--dataset cifar10 \
--data_dir "./../../../data/cifar10" \
--partition_method homo  \
--client_number 8 \
--client_model resnet56 \
--comm_round 200 \
--epochs_client 1 \
--whether_training_on_client 1 \
--epochs_server 20 \
--batch_size 256 \
--optimizer Adam \
--lr 0.001 \
--running_name ResNet56_homo_cifar10
# --multi_gpu_server
