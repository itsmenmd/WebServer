import mysql.connector 
def mysql_conn():
	conn = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
	return conn