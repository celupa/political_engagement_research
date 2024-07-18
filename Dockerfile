# for lambda deployment
FROM public.ecr.aws/lambda/python:3.10

COPY ["lambda_requirements.txt", "poleng_xgb.bin", "lambda_function.py", "./"]

RUN pip install -r lambda_requirements.txt

CMD ["lambda_function.lambda_handler"]
