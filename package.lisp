;;;; package.lisp

(defpackage #:april-utils
  (:use #:cl #:vk)
  (:export
   #:with-mapped-memory
   #:copy-to-device
   #:copy-from-device
   #:buffer-size))

(defpackage #:april-gpu
  (:use #:cl #:vk #:iterate #:access #:april-utils)
  (:local-nicknames (:au :april-utils)))
