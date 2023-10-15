(in-package #:april-utils)

(defmacro with-mapped-memory ((p-data device memory offset size) &body body)
  `(cffi:with-foreign-object (,p-data :pointer)
     (vk:map-memory ,device ,memory ,offset ,size ,p-data)
     (unwind-protect
          (progn ,@body)
       (vk:unmap-memory ,device ,memory))))


(defun copy-to-device (device memory data data-type &optional (offset 0))
  "Copies data to device memory.
DEVICE - a VkDevice handle
MEMORY - a VkDeviceMemory handle
DATA - lisp data to copy
DATA-TYPE - a foreign CFFI type corresponding to DATA's type."
  (let* ((data-count (cond
                       ((listp data) (length data))
                       ((arrayp data) (array-total-size data))
                       (t 1)))
         (data-size (* (cffi:foreign-type-size data-type) data-count)))
    (with-mapped-memory (p-mapped device memory offset data-size)
      (cffi:with-foreign-object (p-data data-type data-count)
        (dotimes (i data-count)
          (setf (cffi:mem-aref p-data data-type i)
                (cond
                  ((arrayp data) (aref data i))
                  ((listp data) (nth i data))
                  (t data))))
        (vk-utils:memcpy (cffi:mem-aref p-mapped :pointer)
                         p-data
                         data-size)))))

(defun copy-from-device (device memory data data-type &optional (offset 0))
  "Copies data from device memory.
DEVICE - a VkDevice handle
MEMORY - a VkDeviceMemory handle
DATA - lisp array to copy to
DATA-TYPE - a foreign CFFI type corresponding to DATA's type."
  (let* ((data-count (cond
                       ((listp data) (length data))
                       ((arrayp data) (array-total-size data))
                       (t 1)))
         (data-size (* (cffi:foreign-type-size data-type) data-count)))
    (with-mapped-memory (p-mapped device memory offset data-size)
      (cffi:with-foreign-object (p-data data-type data-count)
        (dotimes (i data-count)
          (setf (cffi:mem-aref p-data data-type i)
                (cond
                  ((arrayp data) (aref data i))
                  ((listp data) (nth i data))
                  (t data))))
        (vk-utils:memcpy (cffi:mem-aref p-mapped :pointer)
                         p-data
                         data-size)))))

(declaim (inline buffer-size))
(defun buffer-size (buffer type)
  (* (length buffer) (cffi:foreign-type-size type)))
