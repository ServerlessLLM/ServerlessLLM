import os


def is_folder_accessible(folder_path):
    """
    Check if the folder exists and if it's accessible (read, write, execute).

    :param folder_path: The path to the folder.
    :return: True if the folder is accessible, False otherwise.
    """
    # Check if the path exists and is a directory
    if not os.path.isdir(folder_path):
        return False

    # Check for read, write, and execute permissions
    can_read = os.access(folder_path, os.R_OK)
    can_write = os.access(folder_path, os.W_OK)
    can_execute = os.access(folder_path, os.X_OK)

    # The folder is accessible if all permissions are granted
    return can_read and can_write and can_execute


def main():
    home_dir = os.path.expanduser("~")
    model_path = home_dir + "/models"
    if is_folder_accessible(model_path):
        print("The '~/models' folder exists")
    else:
        print(
            "Access issue detected, trying to create the model/ folder in home directory."
        )
        input("Press Enter to continue...")
        os.makedirs(model_path)
        print("The '~/models' folder is successfully created")


if __name__ == "__main__":
    main()
