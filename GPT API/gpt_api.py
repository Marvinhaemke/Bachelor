
from flask import Flask, request, render_template, url_for, abort, send_file, jsonify
from urllib.parse import quote
from bs4 import BeautifulSoup
import os
import shutil

app = Flask(__name__)

# Start Page (Fill out Form, output initial description for CustomGPT and link to selected CustomGPT
@app.route('/')
def hello_world():
    return 'Hello from Flask!'


# Redirect to the apropiate render template file
@app.route('/<page_name>')
def show_page(page_name):
    try:
        return render_template(f'{page_name}/{page_name}.html')
    except:
        abort(404)

# show template
@app.route('/temp')
def show_temp():

    return render_template('colorful-socks/colorful-socks.html')


def create_page(page_name, templates_directory):
    new_directory_path = os.path.join(templates_directory, page_name)

    try:
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)
            return f"Directory {page_name} created successfully."
        else:
            return f"Directory {page_name} already exists."
    except Exception as e:
        return f"An error occurred: {e}"


# Website Designer API Endpoints
@app.route('/write', methods=['POST'])
def write_file():
    content = request.json
    filename = content.get("filename")
    text = content.get("text")
    section_id = content.get("section")

    if not filename or not text:
        return "Invalid request", 400

    # Ensure the filename ends with '.html' for security
    if not filename.endswith('.html'):
        return "Invalid file type. Only HTML files are allowed.", 400

    # Remove '.html' extension and encode the filename for URL
    page_name = quote(filename.rsplit('.html', 1)[0])

    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Append the 'templates' directory to the path
    templates_directory = os.path.join(current_directory, 'templates')

    # Create the new directory if it doesn't exist already
    create_page(page_name, templates_directory)
    new_directory_path = os.path.join(templates_directory, page_name)

    # Construct the full file path
    file_path = os.path.join(new_directory_path, filename)

     # If a section ID is provided, replace only that section's content
    if section_id:
        # Read the existing content and parse with BeautifulSoup
        with open(file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
        section = soup.find(id=section_id)
        if section:
            # Create a new BeautifulSoup object for the new content
            new_section = BeautifulSoup(text, 'html.parser')
            # Replace the old section with the new one
            section.replace_with(new_section)
            # Write the updated content back to the file
            with open(file_path, 'w') as file:
               file.write(str(soup))
            return f"File {filename} updated successfully", 200
        else:
            return f"No section found with id {section_id}", 404
    else:

        with open(file_path, 'w') as file:
            file.write(text)



    # Generate the URL for the newly created file
    file_url = url_for('show_page', page_name=page_name, _external=True)

    return f"File {filename} written successfully. View at {file_url}", 200

@app.route('/writecss', methods=['POST'])
def write_css():
    content = request.json
    filename = content.get("filename")
    text = content.get("text")

    if not filename or not text:
        return "Invalid request", 400

    # Ensure the filename ends with '.css'
    if not filename.endswith('.css'):
        return "Invalid file type. Only CSS files are allowed.", 400

    # Remove .css from filename
    page_name = quote(filename.rsplit('.css', 1)[0])

    # Get the static/css directory
    static_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', page_name, 'css')

    # Ensure the static/css directory exists
    if not os.path.exists(static_directory):
        os.makedirs(static_directory)

    file_path = os.path.join(static_directory, filename)

    with open(file_path, 'a') as file:
        file.write(text)

    return f"CSS file {filename} written successfully", 200



@app.route('/saveImage', methods=['POST'])
def save_image():
    if 'image' not in request.files or 'pagename' not in request.form:
        return jsonify({'error': 'Invalid request, missing image'}), 400

    if 'pagename' not in request.form:
        return jsonify({'error': 'Invalid request, missing pagename'}), 400

    if 'filename' not in request.form:
        return jsonify({'error': 'Invalid request, missing filename'}), 400

    if request.files['image'] == False:
        return jsonify({'error': 'Invalid request, request.files not found'}), 400

    image = request.files['image']
    pagename = request.form['pagename']
    filename = request.form['filename']

    # Get the static/css directory
    static_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', pagename, 'images')

    # Ensure the static/css directory exists
    if not os.path.exists(static_directory):
        os.makedirs(static_directory)

    file_path = os.path.join(static_directory, filename)
    # Save the file
    image.save(file_path)

    return jsonify({'message': 'Image saved successfully', 'url': f'/{static_directory}/{filename}'}), 200


# Template modifier API Endpoints
'''
@app.route('/rewrite', methods=['POST'])
def write_section():
    content = request.json
    page_name = content.get("pagename")
    text = content.get("text")
    section_id = content.get("section")

    if not pagename or not text or not section:
        return "Invalid request! pagename, text or section is missing.", 400

    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Append the 'templates' directory to the path
    templates_directory = os.path.join(current_directory, 'templates')

    # Construct the full directoy path
    directory_path = os.path.join(templates_directory, page_name)

    # Construct the full file path
    file_path = os.path.join(directory_path, filename)

    # Create an empty copy of the selected template at the new file path if it doesn't exist already
    if not os.path.exists(directory_path):
        # Step 1: Copy the directory
        shutil.copytree(source_dir, directory_path)

        # Step 2: Rename the directory
        os.rename(directory_path, page_name)

        #

        # Step 3: Parse and modify the HTML file
        with open(new_file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')

        # Clear the content inside <body>
        if soup.body:
            soup.body.clear()

        # Write the modified HTML back to the file
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(str(soup))

     # If a section ID is provided, replace only that section's content
    if section_id:
        # Read the existing content and parse with BeautifulSoup
        with open(file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
        section = soup.find(id=section_id)
        if section:
            # Create a new BeautifulSoup object for the new content
            new_section = BeautifulSoup(text, 'html.parser')
            # Replace the old section with the new one
            section.replace_with(new_section)
            # Write the updated content back to the file
            with open(file_path, 'w') as file:
               file.write(str(soup))
            return f"File {filename} updated successfully", 200
        else:
            return f"No section found with id {section_id}", 404
    else:

        with open(file_path, 'w') as file:
            file.write(text)



    # Generate the URL for the newly created file
    file_url = url_for('show_page', page_name=page_name, _external=True)

    return f"Section {section_id} written successfully. View at {file_url}", 200
'''
