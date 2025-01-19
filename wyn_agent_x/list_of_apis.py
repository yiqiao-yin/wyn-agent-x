from typing import Any, Dict, List

from serpapi import GoogleSearch
from twilio.rest import Client

from wyn_agent_x.helper import register_function


@register_function("send_sms")
def send_sms(
    payload: Dict[str, str], secrets: Dict[str, str], event_stream: list
) -> Dict[str, Any]:
    """
    Simulate sending an SMS using the Twilio API.

    Args:
        payload (Dict[str, str]): Contains the message details and recipient information.
        secrets (Dict[str, str]): Contains sensitive data like account SID and auth token.
        event_stream (list): A list to log events and responses for the SMS API call.

    Returns:
        Dict[str, Any]: A dictionary containing the status and result of the SMS operation.

    The function limits the message body to 1600 characters.
    If exceeded, it returns an error response without sending the SMS.
    """
    print(f"👀 API Call: Sending SMS with payload: {payload}")

    # Extract Secrets
    account_sid = secrets["account_sid"]
    auth_token = secrets["auth_token"]

    # Message body check for character limit
    message_body = (
        f"Hello {payload['name']}, here's the message: {payload['message body']}"
    )
    if len(message_body) > 1600:
        response = {
            "status": "error",
            "message": "Message body exceeds 1600 character limit.",
            "model_name": "None",
        }
        print("👀 Error: Message body exceeds the 1600 character limit.")

        # Append the error result to the event stream
        event_stream.append(
            {"event": "api_call", "api_name": "send_sms", "response": response}
        )
        return response

    try:
        # Initialize Twilio Client with provided credentials
        client = Client(account_sid, auth_token)

        # Simulate sending a message (replace with actual logic for real SMS sending)
        message = client.messages.create(
            body=message_body,
            from_="+18552060350",  # Replace with a valid Twilio number
            to="+15859538396",  # Replace with the destination number
        )

        print(f"👀 Message SID: {message.sid}")
        response = {"status": f"success: {message.sid}", "model_name": "None"}

    except Exception as e:
        # Gracefully handle any exceptions
        response = {
            "status": f"error: {str(e)}",
            "message": "Failed to send SMS due to an error.",
            "model_name": "None",
        }
        print(f"👀 Error: {str(e)}")

    # Append the result to the event stream
    event_stream.append(
        {"event": "api_call", "api_name": "send_sms", "response": response}
    )

    return response


# Register the google_search function
@register_function("google_search")
def google_search(
    payload: Dict[str, str], secrets: Dict[str, str], event_stream: list
) -> Dict[str, Any]:
    """
    Simulate a Google search using the SerpAPI and return the results formatted as a Markdown table.

    Args:
        payload (Dict[str, str]): A dictionary containing the search parameters:
                                  - query: The search query string.
                                  - location: The location for the search.
        secrets (Dict[str, str]): Contains the SerpAPI key for API authentication.
        event_stream (list): A list used to log events and responses.

    Returns:
        Dict[str, Any]: A dictionary with the status of the API call and a Markdown table of search results.
    """

    # Extract Secrets
    serpapi_key = secrets["serpapi_key"]

    # Extract search parameters from the payload
    engine = payload.get("engine", "google")
    query = payload.get("query", "")
    location = payload.get("location", "")
    num = payload.get("num", "50")
    num = int(num)
    num = num if type(num) == int else 50

    # Checkpoint
    print(f"Run google search API using: \nquery={query}, \nlocation={location}")

    try:
        # Perform the Google Search using SerpAPI
        search = GoogleSearch(
            {
                "engine": engine,
                "q": query,
                "location": location,
                "api_key": serpapi_key,
                "num": num,
            }
        )
        results = search.get_dict()

        # Extract relevant results
        organic_results = results.get("organic_results", [])

        # Convert search results to a Markdown table
        md_table = "| Title | Link | Snippet |\n| :--- | :--- | :--- |\n"
        for item in organic_results:
            title = item.get("title", "N/A")
            link = item.get("link", "#")
            snippet = item.get("snippet", "N/A")
            md_table += f"| {title} | [Link]({link}) | {snippet} |\n"
        print(f"👀 Results: \n{md_table}")

        # Prepare the response
        response = {"status": "success", "model_name": "None", "results": md_table}

        # Append the result to the event stream
        event_stream.append(
            {
                "event": "api_call",
                "api_name": "google_search",
                "response": {"text": response, "status": "200 success"},
            }
        )

    except Exception as e:
        # Handle any exceptions that occur during the API call
        response = {"status": f"error: {str(e)}", "model_name": "None"}
        event_stream.append(
            {"event": "api_call", "api_name": "google_search", "response": response}
        )

    return response


import json  # To format dictionary as text
import os
import smtplib
from email.message import EmailMessage

import matplotlib.pyplot as plt
import pandas as pd  # To handle DataFrame type
import yfinance as yf
from fpdf import FPDF


@register_function("generate_financial_report")
def generate_financial_report(
    payload: Dict[str, str], secrets: Dict[str, str], event_stream: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a financial report for a given ticker and save it as a PDF.

    Args:
        payload (Dict[str, str]): Contains the ticker symbol and save_pdf flag:
                                  - ticker: The stock ticker symbol (e.g., "AAPL").
                                  - save_pdf: Whether to save the report as a PDF (True/False).
        secrets (Dict[str, str]): Not used but included for consistency.
        event_stream (list): A list to log events and responses.

    Returns:
        Dict[str, Any]: A dictionary containing the status and the filename if saved.
    """
    # Extract payload values
    ticker = payload.get("ticker", "AAPL")
    save_pdf = payload.get("save_pdf", True)
    send_email = payload.get("send_email", True)
    to_email = payload.get("to_email", "eagle0504@gmail.com")

    # Extract secrets
    email_key = secrets["email_key"]

    # Log API call details
    print(f"Generating financial report for ticker: {ticker}, save_pdf: {save_pdf}")

    # Fetching data from Yahoo Finance
    result = yf.Ticker(ticker)
    historical_data = result.history(period="10y")
    analyst_price_targets = result.analyst_price_targets
    recommendations_summary = result.recommendations_summary
    info = result.info

    # Extract relevant info for the report
    long_business_summary = info.get("longBusinessSummary", "No summary available.")
    sector = info.get("sector", "No sector information available.")
    website = info.get("website", "No website information available.")

    # Function to format data as text
    def format_data(data):
        if data is None:
            return "No data available."
        elif isinstance(data, pd.DataFrame):
            return data.to_string(index=False)
        elif isinstance(data, dict):
            return json.dumps(data, indent=4)
        else:
            return str(data)

    # Function to calculate RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Function to plot the stock chart
    def plot_stock_chart(data, ticker, filename, analyst_price_targets):
        plt.figure(figsize=(10, 8))

        # Subplot 1: Stock price with moving averages
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data["Close"], label="Close Price", color="blue")
        plt.plot(data.index, data["SMA20"], label="SMA20", color="orange")
        plt.plot(data.index, data["SMA50"], label="SMA50", color="green")
        plt.plot(data.index, data["SMA100"], label="SMA100", color="red")

        # Add analyst price targets as stars
        if analyst_price_targets:
            last_date = data.index[-1]
            next_date = last_date + pd.Timedelta(
                days=1
            )  # Assume the next timestamp is the next day
            plt.scatter(
                next_date,
                analyst_price_targets["current"],
                color="gold",
                label="Current Price Target",
                s=200,
                marker="*",
            )
            plt.scatter(
                next_date,
                analyst_price_targets["low"],
                color="red",
                label="Low Price Target",
                s=200,
                marker="*",
            )
            plt.scatter(
                next_date,
                analyst_price_targets["high"],
                color="green",
                label="High Price Target",
                s=200,
                marker="*",
            )
            plt.scatter(
                next_date,
                analyst_price_targets["mean"],
                color="blue",
                label="Mean Price Target",
                s=200,
                marker="*",
            )
            plt.scatter(
                next_date,
                analyst_price_targets["median"],
                color="purple",
                label="Median Price Target",
                s=200,
                marker="*",
            )

        plt.title(f"{ticker} Stock Price and Moving Averages")
        plt.legend()
        plt.grid()

        # Subplot 2: RSI
        plt.subplot(2, 1, 2)
        plt.plot(data.index, data["RSI"], label="RSI", color="purple")
        plt.axhline(70, color="red", linestyle="--", linewidth=1)
        plt.axhline(30, color="green", linestyle="--", linewidth=1)
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(filename, format="png")  # Save the chart as a PNG file
        plt.close()

    # Add moving averages and RSI to the historical data
    historical_data["SMA20"] = historical_data["Close"].rolling(window=20).mean()
    historical_data["SMA50"] = historical_data["Close"].rolling(window=50).mean()
    historical_data["SMA100"] = historical_data["Close"].rolling(window=100).mean()
    historical_data["RSI"] = calculate_rsi(historical_data["Close"])

    response = {}

    try:
        if save_pdf:
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, f"Financial Summary for {ticker}", ln=True, align="C")

            # Add business overview
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Business Overview", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 10, f"Sector: {sector}")
            pdf.multi_cell(0, 10, f"Website: {website}")
            pdf.multi_cell(0, 10, f"Summary: {long_business_summary}")

            # Plot stock chart and save it as a PNG
            chart_filename = f"{ticker}_stock_chart.png"
            plot_stock_chart(
                historical_data, ticker, chart_filename, analyst_price_targets
            )

            # Add the stock chart to the PDF
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, "Stock Chart", ln=True)
            pdf.image(chart_filename, x=10, y=30, w=180)  # Add the PNG image

            # Save PDF
            pdf_filename = f"{ticker}_report.pdf"
            pdf.output(pdf_filename)
            print(f"PDF saved as {pdf_filename}.")
            response = {"status": "success", "file": pdf_filename}

            if send_email:
                # Param required for sending email
                file_name = pdf_filename
                # to_email = to_email

                # Check if the file is a PDF
                if not file_name.endswith(".pdf"):
                    raise ValueError("The file must be a .pdf file")

                # Check if the file exists
                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File '{file_name}' not found in the local directory"
                    )

                # Email credentials (use environment variables or replace directly)
                from_email = "eagle0504@gmail.com"  # Your email address
                password = email_key  # Your email password (use app-specific password for security)

                # Create email message
                msg = EmailMessage()
                msg["Subject"] = "Sample PDF Report"
                msg["From"] = from_email
                msg["To"] = to_email
                msg.set_content("Please find the attached PDF file.")

                # Attach the PDF file
                with open(file_name, "rb") as f:
                    pdf_data = f.read()
                    msg.add_attachment(
                        pdf_data,
                        maintype="application",
                        subtype="pdf",
                        filename=file_name,
                    )

                # Send the email
                try:
                    with smtplib.SMTP_SSL(
                        "smtp.gmail.com", 465
                    ) as server:  # SMTP server for Gmail
                        server.login(from_email, password)
                        server.send_message(msg)
                    print(
                        f"Email sent successfully to {to_email} with the attached PDF: {file_name}"
                    )
                except Exception as e:
                    print(f"Failed to send email: {str(e)}")
        else:
            response = {"status": "skipped", "message": "PDF saving is disabled."}

    except Exception as e:
        response = {
            "status": f"error: {str(e)}",
            "message": "Failed to generate financial report.",
        }
        print(f"Error: {str(e)}")

    # Append result to event stream
    event_stream.append(
        {
            "event": "api_call",
            "api_name": "generate_financial_report",
            "response": response,
        }
    )

    return response
