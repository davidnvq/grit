
clean:
	find . -type d -name __pycache__ -exec rm -rf {} \+
	find . -type d -name *.pyc -exec rm -rf {} \+ 
	