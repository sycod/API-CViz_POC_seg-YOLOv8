# update .bashrc
# .devcontainer/postCreateCommand.sh
#     apply once at the end of VM creation
#     used in separate file because of quotes in echo command

# activate Python virtual environment on startup
echo "source ~/.venv/bin/activate" >> ~/.bashrc

# aliases
echo "alias ll='ls -alFh'" >> ~/.bashrc
echo "alias cd_project='cd $PWD'" >> ~/.bashrc
