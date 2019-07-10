;; Rebind keys
(global-set-key "\C-d" 'python-shell-send-region)
(global-set-key "\C-q" 'query-replace)
(global-set-key "\C-c\C-e" 'eval-region)
(global-set-key "\M-q" 'whitespace-cleanup)
(global-set-key "\M-m" 'desktop-save)
(global-unset-key "\C-c\C-p")
(global-set-key "\C-c\C-p" 'run-python)
(global-set-key "\M-p" 'previous-line)
(global-set-key "\M-n" 'next-line)
(global-set-key "\C-d" 'projectile-pt) ;; setting for this


;; Finding Files
(defun find-init-file ()
  (interactive)
  (find-alternate-file-other-window user-init-file))
(defun find-custom-file ()
  (interactive)
  (find-alternate-file-other-window
   (expand-file-name "custom.el" prelude-personal-dir)))

(setq dired-listing-switches "-alh")


(require 'swiper)
(global-set-key "\C-s" 'swiper)
(define-key swiper-map (kbd "C-.")
  (lambda () (interactive) (insert (format "\\<%s\\>" (with-ivy-window (thing-at-point 'word))))))
(define-key swiper-map (kbd "M-.")
  (lambda () (interactive) (insert (format "\\<%s\\>" (with-ivy-window (thing-at-point 'symbol))))))

(setq tab-width 4)

(require 'paredit)
(setq paredit-mode 1)

(autoload 'enable-paredit-mode "paredit" t)
(add-hook 'clojure-mode-hook #'enable-paredit-mode)

(setq dumb-jump-mode 1)

(defun connect-linux ()
  (interactive)
  (dired "/user@192.168.1.5:/"))

(require 'org)
(add-hook 'org-shiftup-final-hook 'windmove-up)
(add-hook 'org-shiftdown-final-hook 'windmove-down)
(add-hook 'org-shiftleft-final-hook 'windmove-left)
(add-hook 'org-shiftright-final-hook 'windmove-right)

(require 'ox-latex)
(unless (boundp 'org-latex-classes)
  (setq org-latex-classes nil))
(add-to-list 'org-latex-classes
             '("article"
               "\\documentclass{article}"
               ("\\section{%s}" . "\\section*{%s}")
               ("\\subsection{%s}" . "\\subsection*{%s}")
               ("\\paragraph{%s}" . "\\paragraph*{%s}")
               ("\\subparagraph{%s}" . "\\subparagraph*{%s}")))

(require 'auctex-latexmk)
;(auctex-latexmk-setup)

(setq cider-show-error-buffer nil)
(setq cider-cljs-lein-repl
      "(do (require 'figwheel-sidecar.repl-api)
         (figwheel-sidecar.repl-api/start-figwheel!)
         (figwheel-sidecar.repl-api/cljs-repl))")

(require 'ein)

(require 'multiple-cursors)
(global-set-key (kbd "C-S-c C-S-c") 'mc/edit-lines)


(setq flycheck-global-modes '(not 'python-mode))
(setq prelude-flyspell nil)
(setq projectile-mode nil)

(add-hook 'python-shell (lambda ()
                          (setq-local company-mode nil)))

(yas-global-mode t)

(require 'windmove)

;;;###autoload
(defun buf-move-up ()
  "Swap the current buffer and the buffer above the split.
If there is no split, ie now window above the current one, an
error is signaled."
;;  "Switches between the current buffer, and the buffer above the
;;  split, if possible."
  (interactive)
  (let* ((other-win (windmove-find-other-window 'up))
         (buf-this-buf (window-buffer (selected-window))))
    (if (null other-win)
        (error "No window above this one")
      ;; swap top with this one
      (set-window-buffer (selected-window) (window-buffer other-win))
      ;; move this one to top
      (set-window-buffer other-win buf-this-buf)
      (select-window other-win))))

;;;###autoload
(defun buf-move-down ()
"Swap the current buffer and the buffer under the split.
If there is no split, ie now window under the current one, an
error is signaled."
  (interactive)
  (let* ((other-win (windmove-find-other-window 'down))
	 (buf-this-buf (window-buffer (selected-window))))
    (if (or (null other-win)
            (string-match "^ \\*Minibuf" (buffer-name (window-buffer other-win))))
        (error "No window under this one")
      ;; swap top with this one
      (set-window-buffer (selected-window) (window-buffer other-win))
      ;; move this one to top
      (set-window-buffer other-win buf-this-buf)
      (select-window other-win))))

;;;###autoload
(defun buf-move-left ()
"Swap the current buffer and the buffer on the left of the split.
If there is no split, ie now window on the left of the current
one, an error is signaled."
  (interactive)
  (let* ((other-win (windmove-find-other-window 'left))
	 (buf-this-buf (window-buffer (selected-window))))
    (if (null other-win)
        (error "No left split")
      ;; swap top with this one
      (set-window-buffer (selected-window) (window-buffer other-win))
      ;; move this one to top
      (set-window-buffer other-win buf-this-buf)
      (select-window other-win))))

;;;###autoload
(defun buf-move-right ()
"Swap the current buffer and the buffer on the right of the split.
If there is no split, ie now window on the right of the current
one, an error is signaled."
  (interactive)
  (let* ((other-win (windmove-find-other-window 'right))
	 (buf-this-buf (window-buffer (selected-window))))
    (if (null other-win)
        (error "No right split")
      ;; swap top with this one
      (set-window-buffer (selected-window) (window-buffer other-win))
      ;; move this one to top
      (set-window-buffer other-win buf-this-buf)
      (select-window other-win))))


(provide 'buffer-move)

(defun toggle-window-split ()
  (interactive)
  (if (= (count-windows) 2)
      (let* ((this-win-buffer (window-buffer))
             (next-win-buffer (window-buffer (next-window)))
             (this-win-edges (window-edges (selected-window)))
             (next-win-edges (window-edges (next-window)))
             (this-win-2nd (not (and (<= (car this-win-edges)
                                         (car next-win-edges))
                                     (<= (cadr this-win-edges)
                                         (cadr next-win-edges)))))
             (splitter
              (if (= (car this-win-edges)
                     (car (window-edges (next-window))))
                  'split-window-horizontally
                'split-window-vertically)))
        (delete-other-windows)
        (let ((first-win (selected-window)))
          (funcall splitter)
          (if this-win-2nd (other-window 1))
          (set-window-buffer (selected-window) this-win-buffer)
          (set-window-buffer (next-window) next-win-buffer)
          (select-window first-win)
          (if this-win-2nd (other-window 1))))))

(global-set-key "\C-c\C-t\C-t" 'toggle-window-split)

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(ansi-color-names-vector
   ["#3F3F3F" "#CC9393" "#7F9F7F" "#F0DFAF" "#8CD0D3" "#DC8CC3" "#93E0E3" "#DCDCCC"])
 '(company-quickhelp-color-background "#4F4F4F")
 '(company-quickhelp-color-foreground "#DCDCCC")
 '(conda-env-autoactivate-mode t)
 '(conda-env-home-directory "/Users/lpanda/miniconda3/")
 '(custom-enabled-themes (quote (challenger-deep)))
 '(custom-safe-themes
   (quote
    ("f71859eae71f7f795e734e6e7d178728525008a28c325913f564a42f74042c31" default)))
 '(dumb-jump-mode t)
 '(ein:jupyter-default-server-command "jupyter lab")
 '(fci-rule-color "#383838")
 '(jiralib-url "https://jira.remedypartners.com/jira")
 '(nrepl-message-colors
   (quote
    ("#CC9393" "#DFAF8F" "#F0DFAF" "#7F9F7F" "#BFEBBF" "#93E0E3" "#94BFF3" "#DC8CC3")))
 '(package-selected-packages
   (quote
    (dumb-jump markdown-mode org-jira htmlize latex-extra auctex-latexmk zop-to-char zenburn-theme yasnippet yaml-mode which-key volatile-highlights vkill undo-tree swiper smex smartrep smartparens smart-mode-line rainbow-mode rainbow-delimiters pt paredit-menu paredit-everywhere ov org operate-on-number multiple-cursors move-text magit less-css-mode kibit-helper json-mode imenu-anywhere ido-completing-read+ helm-projectile helm-descbinds helm-ag guru-mode grizzl god-mode gitignore-mode gitconfig-mode git-timemachine gist geiser flycheck flx-ido expand-region exec-path-from-shell elisp-slime-nav ein editorconfig easy-kill discover-my-major diminish diff-hl csv-mode crux conda company-anaconda clojure-cheatsheet challenger-deep-theme browse-kill-ring beacon anzu ace-window)))
 '(pdf-view-midnight-colors (quote ("#DCDCCC" . "#383838")))
 '(python-shell-completion-native-disabled-interpreters (quote ("pypy" "ipython")))
 '(python-shell-interpreter "ipython")
 '(python-shell-interpreter-args "ipython --simple-prompt -i")
 '(python-shell-interpreter-interactive-arg "")
 '(smartparens-global-mode t)
 '(vc-annotate-background "#2B2B2B")
 '(vc-annotate-color-map
   (quote
    ((20 . "#BC8383")
     (40 . "#CC9393")
     (60 . "#DFAF8F")
     (80 . "#D0BF8F")
     (100 . "#E0CF9F")
     (120 . "#F0DFAF")
     (140 . "#5F7F5F")
     (160 . "#7F9F7F")
     (180 . "#8FB28F")
     (200 . "#9FC59F")
     (220 . "#AFD8AF")
     (240 . "#BFEBBF")
     (260 . "#93E0E3")
     (280 . "#6CA0A3")
     (300 . "#7CB8BB")
     (320 . "#8CD0D3")
     (340 . "#94BFF3")
     (360 . "#DC8CC3"))))
 '(vc-annotate-very-old-color "#DC8CC3"))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(org-level-1 ((t (:foreground "#00bfff" :box (:line-width 1 :color "grey75" :style pressed-button) :weight bold))))
 '(org-level-2 ((t (:foreground "#ff1493"))))
 '(org-level-3 ((t (:foreground "#f08080" :underline t)))))
