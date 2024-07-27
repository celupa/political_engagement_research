# for lambda deployment
FROM public.ecr.aws/lambda/python:3.10

COPY ["lambda/lambda_requirements.txt", \
      "misc/poleng_xgb.bin", \
      "lambda/lambda_function.py", \
      "./"]

RUN pip install -r lambda_requirements.txt

CMD [lambda_function.lambda_handler]
