from datetime import datetime
def name_prefix():
	dt = datetime.now()
	return dt.strftime("%Y-%m-%d-%H-%M")
if __name__ == "__main__":
	print("For test, name_prefix is ", name_prefix())
