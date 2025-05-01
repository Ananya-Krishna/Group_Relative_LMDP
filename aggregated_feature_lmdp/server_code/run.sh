
#!/bin/bash
for i in {1..20}
do 
nohup python -u test$i.py > out$i.txt & 
done


for i in {1..9}
do 
nohup python -u ucb$i.py > ucb_out$i.txt & 
done