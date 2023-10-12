;;;; april-gpu.lisp
;;;; Author: John Thingstad
;;;; Creation Date: oct. 2 2023
;;;; Short Description: Setup a compute shader pipeline to allow april to run computations on a GPU.

(in-package #:april-gpu)

;;;-----------------------------------------------------------------------------
;;;
;;; Setup compute pipeline
;;;
;;; Based on: Minimal vulcan compute shader
;;; Author:   İlker Işık aka. necrashter
;;;
;;; Blog entry: https://necrashter.github.io/ceng469/
;;;             project/compute-shaders-intro
;;;
;;; Slides at:  https://docs.google.com/presentation/
;;;             d/1KrDE7a3WT551gjLwa0OO09kgZ71lKJf-6CR6kVVeVAM/
;;;             edit#slide=id.g133da4b698b_0_1059
;;;
;;; Code at:    https://github.com/necrashter/minimal-vulkan-compute-shader
;;;
;;; This code uses the common lisp vulcan interface vk
;;; vk-samples provides example code used to implement the spesifics
;;;
;;;-----------------------------------------------------------------------------

;;--------------------------------------------------
;;  Summary
;;--------------------------------------------------
;;  1. Fill application info
;;  2. Enable validation layer and create instance
;;  3. Physical device
;;  4. Queue family
;;  5. Device
;;  6. Create buffers
;;  7. Query memory types
;;  8. Allocate memory
;;  9. Write data to memory
;; 10. Bind buffers to memory
;; 11. Read shader & Create shader module
;; 12. Descriptor set layout
;; 13. Pipeline layout
;; 14. Pipeline
;; 15. Desciptor pool
;; 16. Descriptor sets
;; 17. Utdate descriptor sets to use our buffers
;; 18. Submit work to GPU
;; 19. Command pool
;; 20. Command buffer
;; 21. Record commands
;; 22. Submit work and wait
;; 23. Read output
;; 24. Cleanup
;;--------------------------------------------------

;;--------------------------------------------------
;; Global constants
;;--------------------------------------------------

(defvar *debug-pipeline* t)
(defvar *step* t)

(defvar *application-name* "April GPU")
(defvar *application-version* (list 1 0 0))
(defvar *engine-name* "vk")
(defvar *engine-version* (list 1 0 0))
(defvar *api-version* vk:+header-version-complete+)

(defparameter *enabled-layers* (list "VK_LAYER_KHRONOS_validation"))

;;--------------------------------------------------
;; Exception hirarchy
;;--------------------------------------------------

(define-condition error-compute-pipeline (error) ())

(define-condition error-create-instance (error-compute-pipeline) ())
(define-condition error-physical-device (error-compute-pipeline) ())
(define-condition error-queue-family (error-compute-pipeline) ())
(define-condition error-create-device (error-compute-pipeline) ())
(define-condition error-create-buffers (error-compute-pipeline) ())
(define-condition error-query-memory-types (error-compute-pipeline) ())
(define-condition error-allocate-memory (error-compute-pipeline) ())
(define-condition error-write-data-to-memory (error-compute-pipeline) ())
(define-condition error-bind-buffers-to-memory (error-compute-pipeline) ())
(define-condition error-read-shader (error-compute-pipeline) ())
(define-condition error-create-shader-module (error-compute-pipeline) ())
(define-condition error-descriptor-set-layout (error-compute-pipeline) ())
(define-condition error-pipeline-layout (error-compute-pipeline) ())
(define-condition error-pipeline-cache (error-compute-pipeline) ())
(define-condition error-pipeline (error-compute-pipeline) ())
(define-condition error-desciptor-pool (error-compute-pipeline) ())
(define-condition error-descriptor-sets (error-compute-pipeline) ())
(define-condition error-utdate-descriptor-sets-to-use-our-buffers (error-compute-pipeline) ())
(define-condition error-submit-work-to-GPU (error-compute-pipeline) ())
(define-condition error-command-pool (error-compute-pipeline) ())
(define-condition error-command-buffer (error-compute-pipeline) ())
(define-condition error-record-commands (error-compute-pipeline) ())
(define-condition error-submit-work-and-wait (error-compute-pipeline) ())
(define-condition error-read-output (error-compute-pipeline) ())

;;--------------------------------------------------
;; compute-pipeline-info
;;
;; A class created to contain the data used by create-compute-shader
;;
;; You need:
;;  shader object 
;;  input array for the initial data
;;  output array result of applying the compute shader 
;;
;;--------------------------------------------------

(defclass compute-pipeline-info ()
  ((a-shader :initarg :shader
	     :accessor cpi-shader)
   (a-input  :initarg :input
	     :accessor cpi-input)
   (a-output :initarg :output
	     :accessor cpi-output)))

(defun make-compute-pipeline-info (&key shader input output)
  (make-instance 'compute-pipeline-info :shader shader :input input :output output))

;;--------------------------------------------------
;; compute-shader-info
;;--------------------------------------------------

(defclass compute-shader-info ()
  ((instance
    :initform nil
    :accessor csi-instance)
   (device
    :initform nil
    :accessor csi-device)
   (physical-device
    :initform nil
    :accessor csi-physical-device)
   (vk:memory-type-index
    :initform nil
    :accessor csi-memory-type-index)
   (input-buffer
    :initform nil
    :accessor csi-input-buffer)
   (output-buffer
    :initform nil
    :accessor csi-output-buffer)
   (in-buffer-memory
    :initform nil
    :accessor csi-in-buffer-memory)
   (out-buffer-memory
    :initform nil
    :accessor csi-out-buffer-memory)
   (descriptor-set
    :initform nil
    :accessor csi-descriptor-set)
   (shader-module
    :initform nil
    :accessor csi-shader-module)
   (pipeline-layout
    :initform nil
    :accessor csi-pipeline-layout)
   (pipeline-cache
    :initform nil
    :accessor csi-pipeline-cache)
   (pipelines
    :initform nil
    :accessor csi-pipelines)))

(defun make-compute-shader-info ()
  (make-instance 'compute-shader-info))

;;--------------------------------------------------
;; Compute pipeline
;;--------------------------------------------------

;; do-read-output
;; do-submit-work-and-wait
;; do-record-commands
;; do-command-buffer
;; do-command-pool
;; do-submit-work-to-GPU
;; do-utdate-descriptor-sets-to-use-our-buffers
;; do-descriptor-sets
;; do-desciptor-pool

(defun do-create-pipeline (shader-info compute-shader-info)
  (declare (ignorable compute-shader-info))
  (let* ((my-pipeline-shader-stage-create-info
	   (vk:make-pipeline-shader-stage-create-info
	    :flags nil
	    :stage :compute
	    :module (csi-shader-module shader-info)
	    :name "main"))
	 (my-compute-pipeline-create-info
	   (vk:make-compute-pipeline-create-info
	    :flags nil
	    :stage my-pipeline-shader-stage-create-info
	    :layout (csi-pipeline-layout shader-info))))
    (multiple-value-bind (my-pipelines result)
	(vk:create-compute-pipelines
	 (csi-device shader-info)
	 (list my-compute-pipeline-create-info)
	 (csi-pipeline-cache shader-info))
      
      (when *debug-pipeline*
	(print my-compute-pipeline-create-info)
	(when *step* (break)))
      (unless (eql result :success)
	(error (make-condition 'error-pipeline)))
      (setf (csi-pipelines shader-info) my-pipelines))
    ))

(defun do-pipeline-layout (shader-info compute-shader-info)
  (let* ((my-pipeline-layout-create-info
	   (vk:make-pipeline-layout-create-info
	    :set-layouts (list (csi-descriptor-set shader-info))))
	 (my-pipeline-layout-info
	   my-pipeline-layout-create-info)
	 (my-pipeline-cache-create-info
	   (vk:make-pipeline-cache-create-info)))
    
    (multiple-value-bind (my-pipeline-layout result-layout)
	(vk:create-pipeline-layout
	 (csi-device shader-info)
	 my-pipeline-layout-info)

      (unless (eql result-layout :success)
	(error (make-condition 'error-pipeline-layout)))
      
      (multiple-value-bind (my-pipeline-cache result-cache)
	  (vk:create-pipeline-cache
	   (csi-device shader-info)
	   my-pipeline-cache-create-info)

	(unless (eql result-cache :success)
	  (error (make-condition 'error-pipeline-cache)))

	(when *debug-pipeline*
	  (print my-pipeline-layout-create-info)
	  (print my-pipeline-cache-create-info)
	  (when *step* (break)))

	(setf (csi-pipeline-cache shader-info) my-pipeline-cache)
	(setf (csi-pipeline-layout shader-info) my-pipeline-layout))))
  (do-create-pipeline shader-info compute-shader-info))

(defun do-descriptor-set-layout (shader-info compute-shader-info)
  (setf (csi-descriptor-set shader-info)
	(vk:create-descriptor-set-layout
	 (csi-device shader-info)
	 (vk:make-descriptor-set-layout-create-info
					;:flags (vk:make-descriptor-set-layout-binding-flags-create-info)
	  :bindings
	  (list (vk:make-descriptor-set-layout-binding
		 :binding 0 :descriptor-type :storage-buffer :descriptor-count 1 :stage-flags :compute)
		(vk:make-descriptor-set-layout-binding
		 :binding 1 :descriptor-type :storage-buffer :descriptor-count 1 :stage-flags :compute)))))
  (do-pipeline-layout shader-info compute-shader-info))

(defun do-create-shader-module (shader-info compute-shader-info)
  (let ((my-shader-info
	  (vk:make-shader-module-create-info :code (cpi-shader compute-shader-info))))
    (multiple-value-bind (my-shader-module result)
	(vk:create-shader-module (csi-device shader-info) my-shader-info)
      (when *debug-pipeline*
	(print my-shader-info)
	(when *step*) (break))
      (unless (eql result :success)
	(error (make-condition 'error-create-shader-module)))
      (setf (csi-shader-module shader-info) my-shader-module))
    (do-descriptor-set-layout shader-info compute-shader-info)))

(defun do-bind-buffers-to-memory (shader-info compute-shader-info)
  (declare (ignorable compute-shader-info))
  (let*((my-in-result
	  (vk:bind-buffer-memory
	   (csi-device shader-info)
	   (csi-input-buffer shader-info)
	   (csi-in-buffer-memory shader-info)
	   0))
	(my-out-result
	  (vk:bind-buffer-memory
	   (csi-device shader-info)
	   (csi-output-buffer shader-info)
	   (csi-out-buffer-memory shader-info)
	   0)))
    (unless (and (eql my-in-result :success) (eql my-out-result :success))
      (error (make-condition 'error-bind-buffers-to-memory))))
  (do-create-shader-module shader-info compute-shader-info))

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

(defun do-write-data-to-memory (shader-info compute-shader-info)
  ;; map-memory writes data from device to host memory
  (let ((my-device (csi-device shader-info))
	(my-c-array (csi-in-buffer-memory shader-info))
	(my-lisp-array (cpi-input compute-shader-info)))
    
    (copy-to-device my-device my-c-array my-lisp-array :float)
    
    (do-bind-buffers-to-memory shader-info compute-shader-info)))

(defun do-allocate-memory (shader-info compute-shader-info)  
  (let* ((my-in-buffer-memory-requirements
	   (vk:get-buffer-memory-requirements
	    (csi-device shader-info)
	    (csi-input-buffer shader-info)))
	 (my-out-buffer-memory-requirements
	   (vk:get-buffer-memory-requirements
	    (csi-device shader-info)
	    (csi-output-buffer shader-info)))
	 (my-in-buffer-allocate-info
	   (vk:make-memory-allocate-info
	    :allocation-size (vk:size my-in-buffer-memory-requirements)
	    :memory-type-index (csi-memory-type-index shader-info)))
	 (my-out-buffer-allocate-info
	   (vk:make-memory-allocate-info
	    :allocation-size (vk:size my-out-buffer-memory-requirements)
	    :memory-type-index (csi-memory-type-index shader-info))))
    (multiple-value-bind (my-in-buffer-memory in-status)
	(vk:allocate-memory (csi-device shader-info) my-in-buffer-allocate-info)
      (multiple-value-bind (my-out-buffer-memory out-status)
	  (vk:allocate-memory (csi-device shader-info) my-out-buffer-allocate-info)
	(when *debug-pipeline*
	  (print my-in-buffer-allocate-info)
	  (print my-out-buffer-allocate-info)
	  (print in-status)
          (print my-in-buffer-memory)
          (print out-status)
          (print my-out-buffer-memory)
	  (when *step* (break)))
	(unless (or (eql in-status :success) (eql my-out-buffer-memory :success))
	  (error (make-condition 'error-allocate-memory)))
        (setf (csi-in-buffer-memory shader-info) my-in-buffer-memory)
        (setf (csi-out-buffer-memory shader-info) my-out-buffer-memory)))
    (do-write-data-to-memory shader-info compute-shader-info)))

(defun do-query-memory-types (shader-info compute-shader-info)
  (let* ((my-device (csi-physical-device shader-info))
	 (my-memory-properties (vk:get-physical-device-memory-properties my-device))
	 (my-memory-types (vk:memory-types my-memory-properties))
	 (my-memory-index
	   (position-if
	    (lambda (my-memory-type)
	      (let ((my-propery-flags (vk:property-flags my-memory-type)))
		(and (not (null my-propery-flags))
		     (member :host-visible my-propery-flags)
		     (member :host-coherent my-propery-flags))))
	    my-memory-types)))
    (when *debug-pipeline*
      (print my-memory-index)
      (print (elt my-memory-types my-memory-index))
      (when *step* (break)))
    (setf (csi-memory-type-index shader-info) my-memory-index))
  (do-allocate-memory shader-info compute-shader-info))

(defun do-create-buffers (shader-info compute-pipeline-info)
  (let ((my-input-buffer-create-info
	  (vk:make-buffer-create-info
	   :size (* (length (cpi-input compute-pipeline-info)) (cffi:foreign-type-size :float))
	   :usage :storage-buffer
	   :sharing-mode :exclusive
	   :queue-family-indices (list 1)))
	(my-output-buffer-create-info
	  (vk:make-buffer-create-info
	   :size (* (length (cpi-output compute-pipeline-info)) (cffi:foreign-type-size :float)) ; ieee754: 4 bytes to a float
	   :usage :storage-buffer
	   :sharing-mode :exclusive
	   :queue-family-indices (list 1))))
    (let ((input-buffer (vk:create-buffer (csi-device shader-info) my-input-buffer-create-info))
	  (output-buffer (vk:create-buffer (csi-device shader-info) my-output-buffer-create-info)))
      (unless (and input-buffer output-buffer)
	(error (make-condition 'error-create-buffers)))
      (when *debug-pipeline*
	(print "Input")
	(print my-input-buffer-create-info)
	(print "Output")
	(print my-output-buffer-create-info)
	(when *step* (break)))
      (setf (csi-input-buffer shader-info) input-buffer)
      (setf (csi-output-buffer shader-info) output-buffer)))
  (do-query-memory-types shader-info compute-pipeline-info))

(defun do-create-physical-device (shader-info compute-pipeline-info)
  (let* ((my-physical-device (first (vk:enumerate-physical-devices (csi-instance shader-info))))
	 (my-queue-family-properties (vk:get-physical-device-queue-family-properties my-physical-device))
	 (my-compute-queue-family-index
	   (position-if (lambda (q)
			  (member :compute (vk:queue-flags q)))
			my-queue-family-properties))
	 (my-queue-priority 0.0)
	 (my-device-queue-create-info (vk:make-device-queue-create-info
				       :queue-family-index my-compute-queue-family-index
				       :queue-priorities (list my-queue-priority)))
	 (my-device-create-info (vk:make-device-create-info
				 :queue-create-infos (list my-device-queue-create-info)
				 :enabled-layer-names *enabled-layers*)))
    (multiple-value-bind (my-device status)
	(vk:create-device my-physical-device my-device-create-info)
      (when *debug-pipeline*
	(print (vk:get-physical-device-properties my-physical-device))
	(print my-queue-family-properties)
	(print my-device-create-info)
	(when *step* (break)))
      (unless (eql status :success)
	(error (make-condition 'error-create-device)))
      (setf (csi-physical-device shader-info) my-physical-device)
      (setf (csi-device shader-info) my-device)
      (do-create-buffers shader-info compute-pipeline-info))))

(defun do-create-instance (shader-info compute-pipeline-info)
  (let* ((my-application-info (vk:make-application-info
			       :application-name *application-name*
			       :application-version (apply #'vk:make-version *application-version*)
			       :engine-name *engine-name*
			       :engine-version (apply #'vk:make-version *engine-version*)
			       :api-version *api-version*))
	 (my-create-info (vk:make-instance-create-info
			  :application-info my-application-info
			  :enabled-layer-names *enabled-layers*)))
    (multiple-value-bind (my-instance status)
	(vk:create-instance my-create-info)
      (when *debug-pipeline*
	(print my-create-info)
	(if *step* (break)))
      (unless (eql status :success)
	(error (make-condition 'error-create-instance)))
      (setf (csi-instance shader-info) my-instance)
      (do-create-physical-device shader-info compute-pipeline-info))))

(defun create-compute-pipeline (compute-pipeline-info)
  "Execute the compute shader in compute-pipeline-info on the GPU.
The input and output arrays are set in compute-pipeline-info.
Returns compute-shader-info (NB! not compute-pipeline-info)."
  (let ((shader-info (make-compute-shader-info)))
    (unwind-protect 
	 (do-create-instance shader-info compute-pipeline-info)
      (cleanup-compute-pipeline shader-info))))

(defun cleanup-compute-pipeline (compute-shader-info)
  (let ((info compute-shader-info))
    ;; info could still be alive when this returns. Remeber the repl * ** and ***
    ;; Therefore set the destroyed objects to nil

    ;; Device.resetCommandPool(CommandPool, vk::CommandPoolResetFlags());
    ;; Device.destroyFence(Fence);
    ;; Device.destroyDescriptorPool(DescriptorPool);
    ;; Device.destroyCommandPool(CommandPool);

    (when (csi-pipelines info)
      (vk:destroy-pipeline (csi-device info) (csi-pipelines info))
      (setf (csi-pipelines info) nil))
    (when (csi-shader-module info)
      (vk:destroy-shader-module (csi-device info) (csi-shader-module info))
      (setf (csi-shader-module info) nil))
    (when (csi-pipeline-layout info)
      (vk:destroy-pipeline-cache (csi-device info) (csi-pipeline-cache info))
      (setf (csi-pipeline-cache info) nil))
    (when (csi-pipeline-layout info)
      (vk:destroy-pipeline-layout (csi-device info) (csi-pipeline-layout info))
      (setf (csi-pipeline-layout info) nil))
    (when (csi-descriptor-set info)
      (vk:destroy-descriptor-set-layout (csi-device info) (csi-descriptor-set info))
      (setf (csi-descriptor-set info) nil))
    (when (csi-in-buffer-memory info)
      (vk:free-memory (csi-device info) (csi-in-buffer-memory info))
      (setf (csi-in-buffer-memory info) nil))
    (when (csi-out-buffer-memory info)
      (vk:free-memory (csi-device info) (csi-out-buffer-memory info))
      (setf (csi-out-buffer-memory info) nil))
    (when (csi-output-buffer info)
      (vk:destroy-buffer (csi-device info) (csi-output-buffer info))
      (setf (csi-output-buffer info) nil))
    (when (csi-input-buffer info)
      (vk:destroy-buffer (csi-device info) (csi-input-buffer info))
      (setf (csi-input-buffer info) nil))
    (when (csi-device info)
      (vk:destroy-device (csi-device info))
      (setf (csi-device info) nil))
    (when (csi-instance info)
      (vk:destroy-instance (csi-instance info))
      (setf (csi-instance info) nil))))

(defun test-compute-pipeline ()
  (let* ((my-input-vector
	   (make-array 10 :element-type 'float
			  :initial-contents
			  (loop for i from 1 to 10 collecting (float i))))
	 (my-output-vector
	   (make-array 10 :element-type 'float :initial-element .0f0))
	 (my-cpu-vector
	   (make-array 10 :element-type 'float
			  :initial-contents
			  (loop for i from 1 to 10 collecting (float (* i i)))))
	 (my-shader-path "./shaders/")
	 (my-shader-file "square.comp.spv")
	 (my-shader-object
	   (vk-utils:read-shader-source
	    (concatenate 'string my-shader-path my-shader-file)))
	 (my-compute-pipeline-info
	   (make-compute-pipeline-info
	    :shader my-shader-object
	    :input my-input-vector
	    :output my-output-vector)))
    (create-compute-pipeline my-compute-pipeline-info)
    (format t "~&Input:         ~{~a~^, ~}~%" (coerce (cpi-input  my-compute-pipeline-info) 'list))
    (format t "Output shader: ~{~a~^, ~}~%" (coerce (cpi-output my-compute-pipeline-info) 'list))
    (format t "Reference CPU: ~{~a~^, ~}~%" (coerce my-cpu-vector 'list))
    (if (equalp (cpi-output my-compute-pipeline-info) my-cpu-vector)
	(format t "Correlation: OK.~%")
	(format t "Correlation: FAIL!~%"))))
