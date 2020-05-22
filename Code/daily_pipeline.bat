SET log_file="D:\Code Projects\Twitter sentiment\Code\Logs\"%date:~-4,4%%date:~-7,2%%date:~-10,2%_scrape.log
call D:\Anaconda\Scripts\activate.bat 2020tweets
call D:\Anaconda\envs\2020tweets\python.exe -i "D:\Code Projects\Twitter sentiment\Code\pipeline.py" > %log_file%
