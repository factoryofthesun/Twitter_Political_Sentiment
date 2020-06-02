SET log_file="D:\Code Projects\Twitter sentiment\Code\Logs\"%date:~-4,4%%date:~-7,2%%date:~-10,2%_scrape.log
SET TWITTER_API_KEY=N6l6Y47sQ0N2rOvL2FfSFloTX
SET TWITTER_API_SECRET_KEY=ZXBh8UwR1V64eHZo1kakSjvlmvDroEdV7ezyndxyfIiB8wzQCl
SET TWITTER_ACCESS_TOKEN=1263822191990206466-n4RyI38ckIoRIMzwcK3BStoHQCpPXr
SET TWITTER_SECRET_ACCESS_TOKEN=le0XjHuKy0VtSBfRPoZGJEq86vLRANp0jiw7UTzWOfQUI
call D:\Anaconda\Scripts\activate.bat 2020tweets
python "D:\Code Projects\Twitter sentiment\Code\pipeline.py" > %log_file%
