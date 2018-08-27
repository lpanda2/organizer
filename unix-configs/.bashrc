# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias dae='emacsclient -t'
alias env='source activate epdevbundling'
alias em='emacs -nw'
alias playenv='source activate otter'
alias findbash='emacs -nw .bashrcy'

cd ../../data/lpanda/
fish
