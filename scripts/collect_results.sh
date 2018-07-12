scp -i ~/.ssh/MyPCKey.pem ubuntu@ec2-18-228-10-119.sa-east-1.compute.amazonaws.com:/home/ubuntu/FD-SHARC/scripts/simulation_resutls.tar.gz C:/Users/Calil/OneDrive/SHARC

scp -i ~/.ssh/MyPCKey.pem C:/Users/Calil/OneDrive/SHARC/Results/2018_06_30_phase_2_ES/parameters/parameters_directories/parameters_directories.zip ubuntu@ec2-18-228-10-119.sa-east-1.compute.amazonaws.com:/home/ubuntu/FD-SHARC/cases

nohup bash -x simulate_all.sh > simulate_all.out 2>&1 &

scp C:/Users/Calil/OneDrive/SHARC/Results/2018_06_30_phase_2_ES/campaigns/campaign_3/parameters/cases_phase2_campaign3.zip calil@172.16.20.76:/home/calil/FD-SHARC/cases

scp calil@172.16.20.76:/home/calil/FD-SHARC/results.tar.gz C:/Users/Calil/OneDrive/SHARC/Results/2018_06_30_phase_2_ES/campaigns/campaign_1

scp C:/Users/Calil/OneDrive/SHARC/Results/2018_06_30_phase_2_ES/campaigns/campaign_2/parameters/parameters_phase2_campaign2.zip calil@172.16.20.76:/home/calil/FD-SHARC/sharc/parameters