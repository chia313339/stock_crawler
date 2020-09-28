#!/bin/bash

start_time=`date +%s`

cd ~/

source /home/chrislee_ml05/venv/bin/activate

cd /home/chrislee_ml05/stock

python step1.py

end_time=`date +%s`

echo execution time was `expr $end_time - $start_time` s.

echo 'Step 1 finish !!!!!'

start_time=`date +%s`

python step2.py

end_time=`date +%s`

echo execution time was `expr $end_time - $start_time` s.

echo 'Step 2 finish !!!!!'