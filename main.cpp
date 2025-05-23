#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_init.h>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <vector>
#include <print>
#include <fstream>
#include <string_view>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
//                              global vars
////////////////////////////////////////////////////////////////////////////////

SDL_Window*              g_window;
VkInstance               g_instance;
VkDebugUtilsMessengerEXT g_debug_messenger;
VkSurfaceKHR             g_surface;
VkPhysicalDevice         g_physical_device;
VkQueue                  g_queue;
uint32_t                 g_queue_family_index;
VkDevice                 g_device;
VkSwapchainKHR           g_swapchain;
VkFormat                 g_swapchain_image_format;
uint32_t                 g_swapchain_image_count;
VkExtent2D               g_swapchain_extent;
std::vector<VkImage>     g_swapchain_images;
std::vector<VkImageView> g_swapchain_image_views;
VkCommandPool            g_command_pool;
VmaAllocator             g_allocator;
VkDescriptorPool         g_descriptor_pool;
VkDescriptorSetLayout    g_descriptor_set_layout;
VkDescriptorSet          g_descriptor_set;

struct Frame
{
  VkCommandBuffer cmd;
  VkFence         fence;
  VkSemaphore     image_available; 
  VkSemaphore     render_finished;
};

std::vector<Frame> g_frames;
uint32_t           g_frame_index = 0;

struct Image
{
  VkImage       handle;
  VkImageView   view;
  VmaAllocation allocation;
  VkFormat      format;
  VkExtent3D    extent;
};

struct Buffer
{
  VkBuffer      handle;
  VmaAllocation allocation;
};

//
// Wavelet Rasterization Resources
//
VkPipeline       g_wr_pipeline;
VkPipelineLayout g_wr_pipeline_layout;
Image            g_wr_image;

////////////////////////////////////////////////////////////////////////////////
//                              misc funcs
////////////////////////////////////////////////////////////////////////////////

inline void exit_if(bool b)
{
  if (b) exit(1);
}

inline void check_vk(VkResult result)
{
  exit_if(result != VK_SUCCESS);
}

auto destroy(Image& image)
{
  assert(image.handle && image.allocation && image.view);
  vkDestroyImageView(g_device, image.view, nullptr);
  vmaDestroyImage(g_allocator, image.handle, image.allocation);
  image = {};
}

auto destroy(Buffer& buffer)
{
  assert(buffer.handle && buffer.allocation);
  vmaDestroyBuffer(g_allocator, buffer.handle, buffer.allocation);
  buffer = {};
}

void release_resources()
{
  vkDeviceWaitIdle(g_device);
  
  // release wavelet rasterization resources
  destroy(g_wr_image);
  vkDestroyPipeline(g_device, g_wr_pipeline, nullptr);
  vkDestroyPipelineLayout(g_device, g_wr_pipeline_layout, nullptr);

  // release other
  vkDestroyDescriptorSetLayout(g_device, g_descriptor_set_layout, nullptr);
  vkDestroyDescriptorPool(g_device, g_descriptor_pool, nullptr);
  vmaDestroyAllocator(g_allocator);
  for (auto& frame : g_frames)
  {
    vkDestroySemaphore(g_device, frame.image_available, nullptr);
    vkDestroySemaphore(g_device, frame.render_finished, nullptr);
    vkDestroyFence(g_device, frame.fence, nullptr);
    vkFreeCommandBuffers(g_device, g_command_pool, 1, &frame.cmd);
  }
  vkDestroyCommandPool(g_device, g_command_pool, nullptr);
  for (auto image_view : g_swapchain_image_views)
    vkDestroyImageView(g_device, image_view, nullptr);
  vkDestroySwapchainKHR(g_device, g_swapchain, nullptr);
  vkDestroyDevice(g_device, nullptr);
  vkDestroySurfaceKHR(g_instance, g_surface, nullptr);
  vkDestroyDebugUtilsMessengerEXT(g_instance, g_debug_messenger, nullptr);
  vkDestroyInstance(g_instance, nullptr);
  SDL_DestroyWindow(g_window);
  SDL_Quit();
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT      message_severity,
  VkDebugUtilsMessageTypeFlagsEXT             message_type,
  VkDebugUtilsMessengerCallbackDataEXT const* callback_data,
  void*                                       user_data)
{
  std::println("{}", callback_data->pMessage);
  return VK_FALSE;
}

auto get_debug_info()
{
  return VkDebugUtilsMessengerCreateInfoEXT
  {
    .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
    .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    .pfnUserCallback = debug_messenger_callback,
  };
  
}

VkResult vkCreateDebugUtilsMessengerEXT(
  VkInstance                                  instance,
  VkDebugUtilsMessengerCreateInfoEXT const*   pCreateInfo,
  VkAllocationCallbacks const*                pAllocator,
  VkDebugUtilsMessengerEXT*                   pMessenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) 
    return func(instance, pCreateInfo, pAllocator, pMessenger);
  return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void vkDestroyDebugUtilsMessengerEXT(
  VkInstance                                  instance,
  VkDebugUtilsMessengerEXT                    messenger,
  VkAllocationCallbacks const*                pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
    func(instance, messenger, pAllocator);
}

auto get_file_data(std::string_view filename)
{
  std::ifstream file(filename.data(), std::ios::ate | std::ios::binary);
  exit_if(!file.is_open());

  auto file_size = (size_t)file.tellg();
  // A SPIR-V module is defined a stream of 32bit words
  auto buffer    = std::vector<uint32_t>(file_size / sizeof(uint32_t));
  
  file.seekg(0);
  file.read((char*)buffer.data(), file_size);

  file.close();
  return buffer;
}

auto create_shader_module(std::string_view filename)
{
  auto data = get_file_data(filename);
  VkShaderModuleCreateInfo shader_info
  {
    .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = data.size() * sizeof(uint32_t),
    .pCode    = reinterpret_cast<uint32_t*>(data.data()),
  };
  VkShaderModule shader_module;
  check_vk(vkCreateShaderModule(g_device, &shader_info, nullptr, &shader_module));
  return shader_module;
}

void transform_image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout)
{
  VkImageMemoryBarrier barrier
  {
    .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    .oldLayout           = old_layout,
    .newLayout           = new_layout,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image               = image,
    .subresourceRange     = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

auto create_image(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage)
{
  Image image
  {
    .format = format,
    .extent = { extent.width, extent.height, 1 },
  };

  VkImageCreateInfo image_info
  {
    .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .imageType   = VK_IMAGE_TYPE_2D,
    .format      = image.format,
    .extent      = image.extent,
    .mipLevels   = 1,
    .arrayLayers = 1,
    .samples     = VK_SAMPLE_COUNT_1_BIT,
    .tiling      = VK_IMAGE_TILING_OPTIMAL,
    .usage       = usage,
  };

  VmaAllocationCreateInfo alloc_info
  {
    .flags         = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    .usage         = VMA_MEMORY_USAGE_AUTO,
  };
  check_vk(vmaCreateImage(g_allocator, &image_info, &alloc_info, &image.handle, &image.allocation, nullptr));

  VkImageViewCreateInfo image_view_info
  {
    .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    .image    = image.handle,
    .viewType = VK_IMAGE_VIEW_TYPE_2D,
    .format   = image.format,
    .subresourceRange =
    {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
    },
  };
  check_vk(vkCreateImageView(g_device, &image_view_info, nullptr, &image.view));

  return image;
}

auto create_buffer(uint32_t size, VkBufferUsageFlags usages, VmaAllocationCreateFlags flags)
{
  Buffer buffer;

  VkBufferCreateInfo buf_info
  {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size  = size,
    .usage = usages,
  };
  VmaAllocationCreateInfo alloc_info
  {
    .flags = flags,
    .usage = VMA_MEMORY_USAGE_AUTO,
  };
  check_vk(vmaCreateBuffer(g_allocator, &buf_info, &alloc_info, &buffer.handle, &buffer.allocation, nullptr));

  return buffer;
}

void blit_image(VkCommandBuffer cmd, VkImage src, VkImage dst, VkExtent2D src_extent, VkExtent2D dst_extent)
{
  VkImageBlit2 blit
  { 
    .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
    .srcSubresource =
    {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .layerCount = 1,
    },
    .dstSubresource =
    {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .layerCount = 1,
    },
  };
  blit.srcOffsets[1].x = src_extent.width;
  blit.srcOffsets[1].y = src_extent.height;
  blit.srcOffsets[1].z = 1;
  blit.dstOffsets[1].x = dst_extent.width;
  blit.dstOffsets[1].y = dst_extent.height;
  blit.dstOffsets[1].z = 1;

  VkBlitImageInfo2 info
  {
    .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
    .srcImage       = src,
    .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .dstImage       = dst,
    .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .regionCount    = 1,
    .pRegions       = &blit,
    .filter         = VK_FILTER_LINEAR,
  };

  vkCmdBlitImage2(cmd, &info);
}

////////////////////////////////////////////////////////////////////////////////
//                              init funcs
////////////////////////////////////////////////////////////////////////////////

void init_SDL()
{
  // SDL init
  SDL_Init(SDL_INIT_VIDEO);

  // create SDL window
  exit_if(!(g_window = SDL_CreateWindow("SMAA Test", 500, 500, SDL_WINDOW_VULKAN)));
}

void create_instance()
{
  // app info
  uint32_t instance_version = VK_API_VERSION_1_0;
  vkEnumerateInstanceVersion(&instance_version);
  VkApplicationInfo app_info
  {
    .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .apiVersion = instance_version,
  };

  // enable validation layer
  const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
  auto debug_info = get_debug_info();
  
  // get extensions
  uint32_t count;
  auto ret = SDL_Vulkan_GetInstanceExtensions(&count);
  auto extensions = std::vector(ret, ret + count);
  extensions.emplace_back("VK_EXT_debug_utils");

  // create instance
  VkInstanceCreateInfo instance_info
  { 
    .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext                   = &debug_info,
    .pApplicationInfo        = &app_info,
    .enabledLayerCount       = 1,
    .ppEnabledLayerNames     = layers,
    .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
    .ppEnabledExtensionNames = extensions.data(),
  };
  check_vk(vkCreateInstance(&instance_info, nullptr, &g_instance));
}

void create_debug_messenger()
{
  auto debug_info = get_debug_info();
  check_vk(vkCreateDebugUtilsMessengerEXT(g_instance, &debug_info, nullptr, &g_debug_messenger));
}

void create_surface()
{
  exit_if(!SDL_Vulkan_CreateSurface(g_window, g_instance, nullptr, &g_surface));
}

void select_physical_device()
{
  uint32_t count;
  vkEnumeratePhysicalDevices(g_instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(g_instance, &count, devices.data());
  g_physical_device = devices[0];
  exit_if(!g_physical_device);
}

void create_device_and_get_graphics_queue()
{
  // get queue family properties
  uint32_t count;
  vkGetPhysicalDeviceQueueFamilyProperties(g_physical_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(g_physical_device, &count, queue_families.data());

  // get graphics queue properties
  auto it = std::find_if(queue_families.begin(), queue_families.end(), [](VkQueueFamilyProperties const& queue_family) 
  {
    return queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT;
  });
  exit_if(it == queue_families.end());

  // set graphics queue info
  auto priority = 1.f;
  g_queue_family_index = static_cast<uint32_t>(std::distance(it, queue_families.begin()));
  VkDeviceQueueCreateInfo queue_info
  {
    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = g_queue_family_index,
    .queueCount       = 1,
    .pQueuePriorities = &priority,
  };

  // features
  VkPhysicalDeviceVulkan13Features features13
  { 
    .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    .pNext               = nullptr, 
    .synchronization2    = true,
    .dynamicRendering    = true,
  };
  VkPhysicalDeviceVulkan12Features features12
  { 
    .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    .pNext               = &features13,
    .bufferDeviceAddress = true,
  };
  VkPhysicalDeviceFeatures2 features2
  {
    .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    .pNext    = &features12,
  };

  // create device
  const char* extensions[] = { "VK_KHR_swapchain" };
  VkDeviceCreateInfo device_info
  {
    .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext                   = &features2,
    .queueCreateInfoCount    = 1,
    .pQueueCreateInfos       = &queue_info,
    .enabledExtensionCount   = 1,
    .ppEnabledExtensionNames = extensions,
  };
  check_vk(vkCreateDevice(g_physical_device, &device_info, nullptr, &g_device));

  // get graphics queue
  vkGetDeviceQueue(g_device, g_queue_family_index, 0, &g_queue);
}

void init_vma()
{
  uint32_t instance_version = VK_API_VERSION_1_0;
  vkEnumerateInstanceVersion(&instance_version);
  VmaAllocatorCreateInfo allocator_info
  {
    .flags            = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT |
                        VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
    .physicalDevice   = g_physical_device,
    .device           = g_device,
    .instance         = g_instance,
    .vulkanApiVersion = instance_version,
  };
  check_vk(vmaCreateAllocator(&allocator_info, &g_allocator));
}

void create_swapchain()
{
  // get surface capabilities
  VkSurfaceCapabilitiesKHR  surface_capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_physical_device, g_surface, &surface_capabilities);

  // get surface formats
  std::vector<VkSurfaceFormatKHR> surface_formats;
  uint32_t count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_physical_device, g_surface, &count, nullptr);
  surface_formats.resize(count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_physical_device, g_surface, &count, surface_formats.data());

  // create swapchain
  VkSwapchainCreateInfoKHR info
  {
    .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    .surface          = g_surface,
    .minImageCount    = surface_capabilities.minImageCount + 1,
    .imageFormat      = surface_formats[0].format,
    .imageColorSpace  = surface_formats[0].colorSpace,
    .imageExtent      = surface_capabilities.currentExtent,
    .imageArrayLayers = 1,
    .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    .preTransform     = surface_capabilities.currentTransform,
    .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .presentMode      = VK_PRESENT_MODE_FIFO_KHR,
    .clipped          = VK_TRUE,
  };
  check_vk(vkCreateSwapchainKHR(g_device, &info, nullptr, &g_swapchain));

  // get swapchain images
  vkGetSwapchainImagesKHR(g_device, g_swapchain, &g_swapchain_image_count, nullptr);
  g_swapchain_images.resize(g_swapchain_image_count);
  vkGetSwapchainImagesKHR(g_device, g_swapchain, &g_swapchain_image_count, g_swapchain_images.data());

  // create image views
  g_swapchain_image_views.resize(g_swapchain_image_count);
  for (size_t i = 0; i < g_swapchain_image_count; ++i)
  {
    VkImageViewCreateInfo info
    {
      .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image            = g_swapchain_images[i],
      .viewType         = VK_IMAGE_VIEW_TYPE_2D,
      .format           = surface_formats[0].format,
      .components       = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    check_vk(vkCreateImageView(g_device, &info, nullptr, &g_swapchain_image_views[i]));
  }

  // set swapchain image format
  g_swapchain_image_format = surface_formats[0].format;
  // get swapchain extent
  g_swapchain_extent = surface_capabilities.currentExtent;
}

void create_command_pool()
{
  VkCommandPoolCreateInfo info
  {
    .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = g_queue_family_index,
  };
  check_vk(vkCreateCommandPool(g_device, &info, nullptr, &g_command_pool));
}

void init_frames()
{
  g_frames.resize(g_swapchain_image_count);
  for (auto& frame : g_frames)
  {
    VkCommandBufferAllocateInfo cmd_info
    {
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = g_command_pool,
      .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount  = 1,
    };
    check_vk(vkAllocateCommandBuffers(g_device, &cmd_info, &frame.cmd));

    VkFenceCreateInfo fence_info
    {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    check_vk(vkCreateFence(g_device, &fence_info, nullptr, &frame.fence));

    VkSemaphoreCreateInfo semaphore_info
    {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    check_vk(vkCreateSemaphore(g_device, &semaphore_info, nullptr, &frame.image_available));
    check_vk(vkCreateSemaphore(g_device, &semaphore_info, nullptr, &frame.render_finished));
  }
}

////////////////////////////////////////////////////////////////////////////////
//                        Wavelet Rasterization Resource Init
////////////////////////////////////////////////////////////////////////////////

void create_descriptor_resources()
{
  // create descriptor pool
  VkDescriptorPoolSize pool_size
  {
    .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    .descriptorCount = 1,
  };
  VkDescriptorPoolCreateInfo pool_info
  {
    .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets       = 1,
    .poolSizeCount = 1,
    .pPoolSizes    = &pool_size,
  };
  check_vk(vkCreateDescriptorPool(g_device, &pool_info, nullptr, &g_descriptor_pool));

  // create descriptor set layout
  VkDescriptorSetLayoutBinding bindings[1]
  {
    { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
  };
  VkDescriptorSetLayoutCreateInfo layout_info
  {
    .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .bindingCount = 1,
    .pBindings    = bindings,
  };
  check_vk(vkCreateDescriptorSetLayout(g_device, &layout_info, nullptr, &g_descriptor_set_layout));

  // allocate descriptor set
  VkDescriptorSetAllocateInfo alloc_info
  {
    .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool     = g_descriptor_pool,
    .descriptorSetCount = 1,
    .pSetLayouts        = &g_descriptor_set_layout,
  };
  check_vk(vkAllocateDescriptorSets(g_device, &alloc_info, &g_descriptor_set));

  // update descriptor set
  std::vector<VkDescriptorImageInfo> image_infos
  {
    { .sampler = VK_NULL_HANDLE, .imageView = g_wr_image.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
  };
  std::vector<VkWriteDescriptorSet> write_infos(image_infos.size());
  for (size_t i = 0; i < image_infos.size(); ++i)
  {
    write_infos[i] = 
    {
      .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet          = g_descriptor_set,
      .dstBinding      = static_cast<uint32_t>(i),
      .descriptorCount = 1,
      .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo      = &image_infos[i],
    };
  }
  vkUpdateDescriptorSets(g_device, static_cast<uint32_t>(write_infos.size()), write_infos.data(), 0, nullptr);
}

void init_wr()
{
  // create image
  g_wr_image = create_image(VK_FORMAT_R32G32B32A32_SFLOAT, g_swapchain_extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  // create descriptor resources
  create_descriptor_resources();

  // create pipeline layout
  VkPipelineLayoutCreateInfo layout_info
  {
    .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts    = &g_descriptor_set_layout,
  };
  check_vk(vkCreatePipelineLayout(g_device, &layout_info, nullptr, &g_wr_pipeline_layout));

  // create compute pipeline
    VkPipelineShaderStageCreateInfo shader_info
  {
    .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = create_shader_module("shader.spv"),
    .pName  = "main",
  };
  VkComputePipelineCreateInfo pipeline_info
  {
    .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage  = shader_info,
    .layout = g_wr_pipeline_layout,
  };
  check_vk(vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &g_wr_pipeline));
  vkDestroyShaderModule(g_device, shader_info.module, nullptr);
}

void init_vk()
{
  // vulkan init
  create_instance();
  create_debug_messenger();
  create_surface();
  select_physical_device();
  create_device_and_get_graphics_queue();
  create_swapchain();
  create_command_pool();
  init_frames();
  init_vma();
  
  // init wavelet rasterization
  init_wr();
}

////////////////////////////////////////////////////////////////////////////////
//                              render funcs
////////////////////////////////////////////////////////////////////////////////

void render()
{
  // get current frame
  auto frame = g_frames[g_frame_index];

  // wait for previous frame
  check_vk(vkWaitForFences(g_device, 1, &frame.fence, VK_TRUE, UINT64_MAX));
  check_vk(vkResetFences(g_device, 1, &frame.fence));

  // acquire next image
  uint32_t image_index;
  check_vk(vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX, frame.image_available, VK_NULL_HANDLE, &image_index));

  // begin command buffer
  check_vk(vkResetCommandBuffer(frame.cmd, 0));
  VkCommandBufferBeginInfo beg_info
  {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(frame.cmd, &beg_info);

  // viewport and scissor
  VkViewport viewport
  {
    .width    = static_cast<float>(g_swapchain_extent.width),
    .height   = static_cast<float>(g_swapchain_extent.height),
    .maxDepth = 1.f
  };
  vkCmdSetViewport(frame.cmd, 0, 1, &viewport);
  VkRect2D scissor{ {}, g_swapchain_extent };
  vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

  transform_image_layout(frame.cmd, g_wr_image.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_wr_pipeline);
  vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_wr_pipeline_layout, 0, 1, &g_descriptor_set, 0, nullptr);
  vkCmdDispatch(frame.cmd, std::ceil((g_swapchain_extent.width + 15) / 16), std::ceil((g_swapchain_extent.height + 15) / 16), 1);

  // copy rendered image to swapchain image
  transform_image_layout(frame.cmd, g_wr_image.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  transform_image_layout(frame.cmd, g_swapchain_images[image_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  blit_image(frame.cmd, g_wr_image.handle, g_swapchain_images[image_index], { g_wr_image.extent.width, g_wr_image.extent.height }, g_swapchain_extent);

  // transform sawpchain image to present layout
  transform_image_layout(frame.cmd, g_swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  // end command buffer
  vkEndCommandBuffer(frame.cmd);

  // submit command
  VkCommandBufferSubmitInfo cmd_submit_info
  {
    .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
    .commandBuffer = frame.cmd,
  };
  VkSemaphoreSubmitInfo wait_sem_submit_info
  {
    .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
    .semaphore = frame.image_available,
    .value     = 1,
    .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
  };
  auto signal_sem_submit_info      = wait_sem_submit_info;
  signal_sem_submit_info.semaphore = frame.render_finished;
  signal_sem_submit_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

  VkSubmitInfo2 submit_info
  {
    .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    .waitSemaphoreInfoCount   = 1,
    .pWaitSemaphoreInfos      = &wait_sem_submit_info,
    .commandBufferInfoCount   = 1,
    .pCommandBufferInfos      = &cmd_submit_info,
    .signalSemaphoreInfoCount = 1,
    .pSignalSemaphoreInfos    = &signal_sem_submit_info,
  };
  check_vk(vkQueueSubmit2(g_queue, 1, &submit_info, frame.fence));

  // present
  VkPresentInfoKHR present_info
  {
    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    .waitSemaphoreCount = 1,
    .pWaitSemaphores    = &frame.render_finished,
    .swapchainCount     = 1,
    .pSwapchains        = &g_swapchain,
    .pImageIndices      = &image_index,
  };
  check_vk(vkQueuePresentKHR(g_queue, &present_info)); 

  // next frame
  g_frame_index = (g_frame_index + 1) % g_swapchain_image_count;
}

////////////////////////////////////////////////////////////////////////////////
//                              main func
////////////////////////////////////////////////////////////////////////////////

int main()
{
  init_SDL();
  init_vk();

  bool quit = false;
  while (!quit)
  {
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
      if (event.type == SDL_EVENT_QUIT)
        quit = true;
    }

    render();
  }

  release_resources();
  
  return 0;
}