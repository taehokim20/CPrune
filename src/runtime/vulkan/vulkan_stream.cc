/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanStream::VulkanStream(const VulkanContext* vctx)
    : vctx_(vctx), state_(new VulkanStreamState()) {
  // create command pool
  VkCommandPoolCreateInfo cmd_pool_cinfo;
  cmd_pool_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmd_pool_cinfo.pNext = nullptr;
  cmd_pool_cinfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cmd_pool_cinfo.queueFamilyIndex = vctx_->queue_family_index;
  VULKAN_CALL(vkCreateCommandPool(vctx_->device, &cmd_pool_cinfo, nullptr, &cmd_pool_));

  VkCommandBufferAllocateInfo buffer_alloc_info;
  buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  buffer_alloc_info.pNext = nullptr;
  buffer_alloc_info.commandPool = cmd_pool_;
  buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  buffer_alloc_info.commandBufferCount = 1;
  VULKAN_CALL(vkAllocateCommandBuffers(vctx_->device, &buffer_alloc_info, &(state_->cmd_buffer_)));

  VkFenceCreateInfo fence_cinfo;
  fence_cinfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_cinfo.pNext = nullptr;
  fence_cinfo.flags = 0;  // VK_FENCE_CREATE_SIGNALED_BIT;
  VULKAN_CALL(vkCreateFence(vctx_->device, &fence_cinfo, nullptr, &(state_->fence_)));

  VkCommandBufferBeginInfo cb_begin;
  cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cb_begin.pNext = nullptr;
  cb_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  cb_begin.pInheritanceInfo = 0;
  VULKAN_CALL(vkBeginCommandBuffer(state_->cmd_buffer_, &cb_begin));
}

VulkanStream::~VulkanStream() {
  vkDestroyFence(vctx_->device, state_->fence_, nullptr);
  vkDestroyCommandPool(vctx_->device, cmd_pool_, nullptr);
}

void VulkanStream::Launch(const std::function<void(VulkanStreamState*)>& kernel) {
  if (vctx_->UseImmediate()) {
    kernel(state_.get());
  } else {
    deferred_kernels_.push_back(kernel);
  }
}

void VulkanStream::LaunchDeferred(const std::function<void()>& deferred_initializer,
                                  const std::function<void(VulkanStreamState*)>& deferred_kernel,
                                  const VulkanStreamToken& deferred_token) {
  ICHECK(!vctx_->UseImmediate());

  // If the new kernel uses the same descriptor set as one of the
  // kernels already in the command buffer, we need to synchronize
  // first.
  if (std::any_of(deferred_tokens_[deferred_token.descriptor_set_].begin(),
                  deferred_tokens_[deferred_token.descriptor_set_].end(),
                  [&](const VulkanStreamToken& token) {
                    DCHECK(token.descriptor_set_ == deferred_token.descriptor_set_);
                    return token.descriptor_set_ == deferred_token.descriptor_set_ &&
                           token.buffers_ != deferred_token.buffers_;
                  })) {
    Synchronize();
  }

  // If the new kernel uses the same buffers in the same descriptor
  // set as an already-queued kernel, we don't need to initialize it
  // again.  Since every VulkanWrappedFunc owns a single descriptor
  // set, unless the same function is called with the same buffer
  // arguments, deferred_initializer() will always be called.
  if (!std::any_of(deferred_tokens_[deferred_token.descriptor_set_].begin(),
                   deferred_tokens_[deferred_token.descriptor_set_].end(),
                   [&](const VulkanStreamToken& token) {
                     DCHECK(token.descriptor_set_ == deferred_token.descriptor_set_);
                     return token.descriptor_set_ == deferred_token.descriptor_set_ &&
                            token.buffers_ == deferred_token.buffers_;
                   })) {
    deferred_initializer();
  }

  // Save the kernel itself to be called later.
  deferred_kernels_.push_back(deferred_kernel);
  deferred_tokens_[deferred_token.descriptor_set_].push_back(deferred_token);
}

void VulkanStream::Synchronize() {
  if (!vctx_->UseImmediate()) {
    for (const auto& deferred_kernel : deferred_kernels_) {
      deferred_kernel(state_.get());
    }
    deferred_kernels_.clear();
    deferred_tokens_.clear();
  } else {
    DCHECK_EQ(deferred_kernels_.size(), 0);
    DCHECK_EQ(deferred_tokens_.size(), 0);
  }

  VULKAN_CALL(vkEndCommandBuffer(state_->cmd_buffer_));
  VkSubmitInfo cb_submit;
  cb_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  cb_submit.pNext = nullptr;
  cb_submit.waitSemaphoreCount = 0;
  cb_submit.pWaitSemaphores = nullptr;
  cb_submit.pWaitDstStageMask = 0;
  cb_submit.commandBufferCount = 1;
  cb_submit.pCommandBuffers = &(state_->cmd_buffer_);
  cb_submit.signalSemaphoreCount = 0;
  cb_submit.pSignalSemaphores = nullptr;

  {
    // Multiple streams (on different threads) use the same VulkanContext
    // instance, so we need to externally synchronize accesses.
    std::lock_guard<std::mutex> g(*(vctx_->queue_mutex));
    VULKAN_CALL(vkQueueSubmit(vctx_->queue, 1, &cb_submit, state_->fence_));
  }
  uint64_t timeout = 1UL << 30UL;
  VkResult res;
  do {
    res = vkWaitForFences(vctx_->device, 1, &(state_->fence_), 0, timeout);
  } while (res == VK_TIMEOUT);
  VULKAN_CHECK_ERROR(res);
  VULKAN_CALL(vkResetCommandBuffer(state_->cmd_buffer_, 0));
  VULKAN_CALL(vkResetFences(vctx_->device, 1, &(state_->fence_)));

  // Re-initialize the command buffer
  VkCommandBufferBeginInfo cb_begin;
  cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cb_begin.pNext = nullptr;
  cb_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  cb_begin.pInheritanceInfo = 0;
  VULKAN_CALL(vkBeginCommandBuffer(state_->cmd_buffer_, &cb_begin));
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
